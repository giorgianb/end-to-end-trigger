#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


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
from numpy.linalg import inv
import sklearn.metrics as metrics
from datasets import get_data_loaders
from tqdm import tqdm
import glob
from torch_geometric.data import Data
import dataclasses
from disjoint_set import DisjointSet
from typing import Union
from numpy.linalg import inv
from scipy.stats import mode


# In[3]:


@dataclasses.dataclass
class EventInfo:
    n_pixels: Union[np.ndarray, torch.Tensor]
    energy: Union[np.ndarray, torch.Tensor]
    momentum: Union[np.ndarray, torch.Tensor]
    interaction_point: Union[np.ndarray, torch.Tensor]
    trigger: Union[bool, torch.Tensor]
    has_trigger_pair: Union[bool, torch.Tensor]
    track_origin: Union[np.ndarray, torch.Tensor]
    trigger_node: Union[np.ndarray, torch.Tensor]
    particle_id: Union[np.ndarray, torch.Tensor]
    particle_type: Union[np.ndarray, torch.Tensor]
    parent_particle_type: Union[np.ndarray, torch.Tensor]
    track_hits: Union[np.ndarray, torch.Tensor]
    track_n_hits: Union[np.ndarray, torch.Tensor]



def get_tracks(edge_index):
    # Get connected components
    ds = DisjointSet()
    for i in range(edge_index.shape[1]):
        ds.union(edge_index[0, i], edge_index[1, i])

    return tuple(list(x) for x in ds.itersets())

def load_graph(filename, min_edge_probability, intt_required=False):
    layers = [(0,), (1,), (2,), (3,4), (5,6)]
    with np.load(filename, allow_pickle=True) as f:
        #model_edge_probability = f['model_edge_probability']
        # TODO: change  back
        f = {k:v for k, v in f.items()}
        if True:
            n_hits = f['hit_cartesian'].shape[0]
            hit_ids = np.arange(n_hits)
            edge_index = np.stack(np.meshgrid(hit_ids, hit_ids), axis=0).reshape(2, -1)
            start, end = edge_index
            pid = f['particle_id']
            y = pid[start] == pid[end]
            edge_index = edge_index[:, y]
    
        else:
            edge_index = f['edge_index'][:, model_edge_probability >= min_edge_probability]
        tracks = get_tracks(edge_index)
        if intt_required:
            tracks = [track for track in tracks if np.any(f['layer_id'][track] >= 3)]
        if 'trigger_node' not in f:
            f['trigger_node'] = np.zeros(f['hit_cartesian'].shape[0])
        if 'has_trigger_pair' not in f:
            f['has_trigger_pair'] = f['trigger']


        track_hits = np.zeros((len(tracks), 3*len(layers)))
        n_pixels = np.zeros((len(tracks), len(layers)))
        energy = np.zeros(len(tracks))
        momentum = np.zeros((len(tracks), 3))
        track_origin = np.zeros((len(tracks), 3))
        trigger_node = np.zeros(len(tracks))
        particle_id = np.zeros(len(tracks))
        particle_type = np.zeros(len(tracks))
        parent_particle_type = np.zeros(len(tracks))
        track_n_hits = np.zeros((len(tracks), len(layers)))

        for i, track in enumerate(tracks):
            layer_id = f['layer_id'][track]
            hit_n_pixels = f['n_pixels'][track]
            hits = f['hit_cartesian'][track]

            # Calculate per-layer information
            for j, layer in enumerate(layers):
                mask = np.isin(layer_id, layer)
                weighted_hits = hit_n_pixels[mask, None] * hits[mask]
                d = np.sum(hit_n_pixels[mask])

                track_hits[i, 3*j:3*(j+1)] = np.sum(weighted_hits, axis=0)/(d + (d == 0))
                n_pixels[i, j] = d
                track_n_hits[i, j] = np.sum(mask)
            
            # Find the GT particle that this track is assigned to
            pids = f['particle_id'][track]
            particle_id[i] = mode(pids, axis=0, keepdims=False).mode
            if np.isnan(particle_id[i]):
                index = track[np.where(np.isnan(pids))[0][0]]
            else:
                index = track[np.where(pids == particle_id[i])[0][0]]

            energy[i] = f['energy'][index]
            momentum[i] = f['momentum'][index]
            track_origin[i] = f['track_origin'][index]
            trigger_node[i] = f['trigger_node'][index]
            particle_type[i] = f['particle_type'][index]
            parent_particle_type[i] = f['parent_particle_type'][index]

        return EventInfo(
                n_pixels=n_pixels,
                energy=energy,
                momentum=momentum,
                interaction_point=f['interaction_point'],
                trigger=f['trigger'],
                has_trigger_pair=f['has_trigger_pair'],
                track_origin=track_origin,
                trigger_node=trigger_node,
                particle_id=particle_id,
                particle_type=particle_type,
                parent_particle_type=parent_particle_type,
                track_hits=track_hits,
                track_n_hits=track_n_hits
        )

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

def matmul_3D(A, B):
    return np.einsum('lij,ljk->lik', A, B)


def get_approximate_radii(track_hits, good_layers, n_layers):
    x_indices = [3*j for j in range(5)]
    y_indices = [3*j+1 for j in range(5)]
    r = np.zeros(track_hits.shape[0])
    centers = np.zeros((track_hits.shape[0], 2))
    for n_layer in range(3, 5 + 1):
        complete_tracks = track_hits[n_layers == n_layer]
        hit_indices = good_layers[n_layers == n_layer]
        if complete_tracks.shape[0] == 0:
            continue

        A = np.ones((complete_tracks.shape[0], n_layer, 3))
        x_values = complete_tracks[:, x_indices]
        x_values = x_values[hit_indices].reshape(complete_tracks.shape[0], n_layer)

        y_values = complete_tracks[:, y_indices]
        y_values = y_values[hit_indices].reshape(complete_tracks.shape[0], n_layer)
        A[:, :, 0] = x_values
        A[:, :, 1] = y_values

        y = - x_values**2 - y_values**2
        y = y.reshape((y.shape[0], y.shape[1], 1))
        AT = np.transpose(A, axes=(0, 2, 1))
        c = matmul_3D(matmul_3D(inv(matmul_3D(AT, A)), AT), y)[..., 0]
        r[n_layers == n_layer] = np.sqrt(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])/200
        centers[n_layers == n_layer] = np.stack([-c[:, 0]/2, -c[:, 1]/2], axis=-1)

    #test = get_approximate_radius(track_hits, n_layers == 5)
    #assert np.allclose(test, r[n_layers == 5])

    return r, centers

def get_length(start, end):
    return np.sqrt(np.sum((start - end)**2, axis=1))


# In[4]:

CONFIG_FILE_PATH = ''
MODEL_FILE_PATH = ''

train_data_loader, valid_data_loader = get_data_laoders(**dconfig)


trigger_files = glob.glob('/ssd3/giorgian/hits-data-january-2024-yasser/trigger/events/*.npz')
nontrigger_files = glob.glob('/ssd3/giorgian/hits-data-january-2024-yasser/nontrigger/events/*.npz')
trigger_output_dir = '/disks/disk1/giorgian/beautyllm-january-2024-npz/trigger/'
nontrigger_output_dir = '/disks/disk1/giorgian/beautyllm-january-2024-npz/nontrigger/'

output_dirs = (trigger_output_dir, nontrigger_output_dir)

for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)
    
all_files = nontrigger_files + trigger_files
trigger_files = set(trigger_files)
nontrigger_files = set(nontrigger_files)

cylindrical_features_scale=np.array([3, 1, 3])
# Loop over each file
for filename in tqdm(all_files):
    # Load the graph data
    event_info = load_graph(
        filename,
        0.5,
        intt_required=True
    )   

    good_layers = np.any(event_info.track_hits.reshape(-1, 5, 3) != 0, axis=-1)
    n_layers = np.sum(good_layers, axis=-1)
    ip = tuple(float(x) for x in event_info.interaction_point)
    radii, centers = get_approximate_radii(event_info.track_hits, good_layers, n_layers)

    p_z = get_predicted_pz(event_info.track_hits,
        good_layers, 
        radii
    )

    parent_ptypes = event_info.parent_ptypes
    gparent_ptypes = event_info.gparent_ptypes



   
    if filename in trigger_files:
        trigger = True
        output_file = os.path.join(trigger_output_dir, os.path.basename(filename))
    else:
        trigger = False
        output_file = os.path.join(nontrigger_output_dir, os.path.basename(filename))

    if False:
        output_file = output_file.replace('.npz', '.txt')
        with open(output_file, 'w') as fout:
            tracks = event_info.track_hits
            print(f'Here is a particle collision event with {len(tracks)} tracks.', file=fout)
            print(f'The collision vertex is {tuple(ip)}.', file=fout)
           # print(f'The collision vertex is {tuple(pred_ip.tolist())}.', file=fout)
            #print(f'{radii=} {p_z=} {centers=} {tracks=}')
            #print(f'{radii.shape=} {p_z.shape=} {centers.shape=} {tracks.shape=}')
            for i, ti in enumerate(np.random.permutation(tracks.shape[0])):
                print(f'Track number {i+1} has a transverse momentum of {radii[ti]}, a parallel momentum of {p_z[ti]}, a center of {tuple(centers[ti].tolist())} and a trajectory of {tuple(tracks[ti].tolist())} as the particle flew through the detector.', file=fout)
    else:
        tracks = event_info.track_hits
        shuffle = np.random.permutation(tracks.shape[0])
        np.savez(
                output_file,
                collision_vertex=ip,
                tracks=tracks[shuffle],
                radii=radii[shuffle],
                p_z=p_z[shuffle],
                centers=centers[shuffle],
                trigger=trigger,
                parent_ptypes=parent_ptypes[shuffle],
                gparent_ptypes=gparent_ptypes[shuffle]
        )




# # 
