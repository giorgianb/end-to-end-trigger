#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dataclasses import replace
import numpy as np
import os
import torch
import os.path
import sys
import logging
import pickle
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from models.garnet_ip import GNNGraphClassifier
from numpy.linalg import inv
import sklearn.metrics as metrics
from datasets import get_data_loaders
from tqdm import tqdm
import glob
from datasets.hit_graph_trigger_pileup import load_graph
from torch_geometric.data import Data


# In[2]:


DEVICE = "cuda:0"


# In[3]:


model_result_folder = '../trigger_results/agnn/agnn-lr9.77987556971304e-05-b64-d16-PReLU-gi1-ln-True-n500000/experiment_2024-09-05_16:06:22/'
config_file = model_result_folder + '/config.pkl'
config = pickle.load(open(config_file, 'rb'))
data_config = config.get('data')
dphi_max, dz_max = data_config['phi_slope_max'], data_config['z0_max']

model_config = config.get('model', {})
model_config.pop('loss_func')
model_config.pop('name')
model = GNNGraphClassifier(**model_config).to(DEVICE)

def load_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer
    return model

# load_checkpoint
checkpoint_dir = os.path.join(model_result_folder, 'checkpoints')
checkpoint_file = sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('model_checkpoint')])
checkpoint_file = checkpoint_file[-1]
print(checkpoint_file)
model = load_checkpoint(checkpoint_file, model)
print('Successfully reloaded!')


# In[4]:


def fit_circles_to_particles(hits, particle_ids):
    """
    Fits a circle to each particle's hits using only the x and y coordinates.

    Parameters:
    hits (numpy.ndarray): An array of shape (n_hits, 3) containing hit coordinates (x, y, z).
    particle_ids (numpy.ndarray): An array of shape (n_hits,) containing particle IDs corresponding to each hit.

    Returns:
    dict: A dictionary of centers keyed by particle ID.
    dict: A dictionary of radii keyed by particle ID.
    """
    unique_pids = np.unique(particle_ids)
    centers = {}
    radii = {}

    for pid in unique_pids:
        # Extract indices of hits corresponding to the current particle ID
        indices = np.where(particle_ids == pid)[0]
        # Extract x and y coordinates; ignore z
        x = hits[indices, 0]
        y = hits[indices, 1]

        if len(x) < 3:
            # Cannot fit a circle with less than 3 points
            continue

        # Set up the linear system for circle fitting in the x-y plane
        D = np.column_stack((x, y, np.ones_like(x)))
        RHS = x**2 + y**2

        # Solve D * params = -RHS to find the circle parameters
        params, residuals, rank, s = np.linalg.lstsq(D, -RHS, rcond=None)
        A, B, C = params
        # Calculate the circle center coordinates
        x0 = -A / 2
        y0 = -B / 2
        # Calculate the radius squared
        R_squared = x0**2 + y0**2 - C

        if R_squared < 0:
            R = np.nan  # Invalid radius (due to numerical issues)
        else:
            R = np.sqrt(R_squared)

        centers[pid] = (x0, y0)
        radii[pid] = R

    return centers, radii

def create_particle_tracks(hits, particle_ids, raw_layers):
    """
    Creates a track for each particle by pooling hits per layer and computing the mean position.

    Parameters:
    hits (numpy.ndarray): An array of shape (n_hits, 3) containing hit coordinates (x, y, z).
    particle_ids (numpy.ndarray): An array of shape (n_hits,) containing particle IDs corresponding to each hit.
    raw_layers (numpy.ndarray): An array of shape (n_hits,) containing raw layer indices for each hit.

    Returns:
    dict: A dictionary mapping particle IDs to their track arrays of shape (15,).
    """
    # Mapping from raw layers to true layers
    raw_layer_to_layer = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4}
    # Vectorize the mapping for efficiency
    vectorized_mapping = np.vectorize(raw_layer_to_layer.get)
    # Map raw layers to true layers
    layers = vectorized_mapping(raw_layers)
    
    # Unique particle IDs
    unique_pids = np.unique(particle_ids)
    tracks = {}

    for pid in unique_pids:
        # Indices of hits corresponding to the current particle
        indices = np.where(particle_ids == pid)[0]
        particle_hits = hits[indices]  # (n_hits_particle, 3)
        particle_layers = layers[indices]  # (n_hits_particle,)

        # Initialize track array with NaNs for missing layers
        track = np.full((5, 3), np.nan)

        for layer in range(5):
            # Indices of hits in the current layer
            layer_indices = np.where(particle_layers == layer)[0]
            if len(layer_indices) > 0:
                # Hits in the current layer
                hits_in_layer = particle_hits[layer_indices]
                # Mean position of hits in the current layer
                mean_position = np.mean(hits_in_layer, axis=0)
                # Store mean position in the track
                track[layer] = mean_position
            # If no hits in the layer, leave NaNs

        # Flatten the track to shape (15,)
        track_flat = track.flatten()
        # Store the track in the dictionary
        tracks[pid] = track_flat

    return tracks

def get_track_endpoints(hits, good_layers):
    # Assumption: all tracks have at least 1 hit
    # If it has one hit, first_hit == last_hit for that track
    # hits shape: (n_tracks, 5, 3)
    # good_layers shape: (n_tracks, 5)
    min_indices = good_layers * np.arange(5) + (1 - good_layers) * np.arange(5, 10)
    indices = np.expand_dims(np.argmin(min_indices, axis=-1), -1)
    indices = np.expand_dims(indices, axis=-2)
    first_hits = np.take_along_axis(hits, indices, axis=-2)
    max_indices = good_layers * np.arange(5, 10) + (1 - good_layers) * np.arange(5)
    indices = np.expand_dims(np.argmax(max_indices, axis=-1), -1)
    indices = np.expand_dims(indices, axis=-2)
    last_hits = np.take_along_axis(hits, indices, axis=-2)
    return first_hits.squeeze(1), last_hits.squeeze(1)

def get_predicted_pz(track_hits, good_layers, radius):
    hits = track_hits.reshape(-1, 5, 3)
    first_hit, last_hit = get_track_endpoints(hits, good_layers)
    dz = (last_hit[:, -1] - first_hit[:, -1])/100
    chord2 = ((last_hit[:, 0] - first_hit[:, 0]) ** 2 + (last_hit[:, 1] - first_hit[:, 1]) ** 2) / 10000
    r2 = 2*radius**2
    with np.errstate(invalid='ignore'):
        dtheta = np.arccos((r2 - chord2) / (r2 + (r2 == 0)))
    dtheta += (dtheta == 0)
    return np.nan_to_num(dz / dtheta)


# In[5]:


trigger_files = [] #glob.glob('/secondssd/giorgian/hits-data-august-2024/trigger/1/*.npz')
nontrigger_files = glob.glob('/secondssd/giorgian/hits-data-august-2024/nontrigger/1/*.npz')
trigger_output_dir = '/home/giorgian/beautyllm/trigger/'
nontrigger_output_dir = '/home/giorgian/beautyllm/nontrigger/'

output_dirs = (trigger_output_dir, nontrigger_output_dir)

for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)
    
all_files = trigger_files + nontrigger_files
cylindrical_features_scale=np.array([3, 1, 3])
# Loop over each file
for filename in tqdm(all_files):
    # Load the graph data
    x, edge_index, y, event_info = load_graph(
        filename,
        cylindrical_features_scale,
        0,
        0,
        use_intt=True,
        construct_edges=False,
        drop_l1=False,
        drop_l2=False,
        drop_l3=False,
        add_global_node=False
    )   
    
    batch = np.zeros(x.shape[0])

    # Create a Data object for PyTorch Geometric
    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        batch=torch.tensor(batch, dtype=torch.long)
    )

    # Send data to the device
    data = data.to(DEVICE)

    # Perform inference
    with torch.no_grad():
        pred = model(data).detach().cpu().numpy()

    pred_ip = pred[0][:2]



    f = np.load(filename)
    pids = np.unique(f['particle_id'])
    tracks = create_particle_tracks(f['hit_cartesian'], f['particle_id'], f['layer_id'])
    centers, radii = fit_circles_to_particles(f['hit_cartesian'], f['particle_id'])
    
    tracks = np.stack([tracks[pid] for pid in pids], axis=0)
    tracks[np.isnan(tracks)] = 0
    
    centers = np.stack([centers[pid] if pid in centers else np.array([0, 0]) for pid in pids], axis=0)
    radii = np.array([radii[pid] if pid in radii else 0 for pid in pids])
    good_layers = np.any(tracks.reshape(-1, 5, 3), axis=-1)
    p_z = get_predicted_pz(tracks, good_layers, radii)

    if 'event1' in filename:
        output_file = os.path.join(trigger_output_dir, os.path.basename(filename))
    else:
        output_file = os.path.join(nontrigger_output_dir, os.path.basename(filename))
    output_file = output_file.replace('.npz', '.txt')

    with open(output_file, 'w') as fout:
        print(f'Here is a particle collision event with {len(tracks)} tracks.', file=fout)
        print(f'The collision vertex is {tuple(pred_ip.tolist())}.', file=fout)
    
        for i, ti in enumerate(np.random.permutation(tracks.shape[0])):
            print(f'Track number {i+1} has a transverse momentum of {radii[ti]}, a parallel momentum of {p_z[ti]}, a center of {tuple(centers[ti].tolist())} and a trajectory of {tuple(tracks[ti].tolist())} as the particle flew through the detector.', file=fout)


