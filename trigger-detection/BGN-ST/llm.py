from dataloaders import get_data_loaders
import tqdm
import os.path
import os
from joblib import Parallel, delayed 
import numpy as np

TRIGGER_OUTPUT_DIR = '/ssd5/giorgian/beauty-llm-pileup-20-trigger/trigger/'
NONTRIGGER_OUTPUT_DIR = '/ssd5/giorgian/beauty-llm-pileup-20-trigger/nontrigger/'
data_config = {'batch_size': 1, 'n_test': 0, 'n_train': 50200, 'n_valid': 0, 'n_workers': 16, 'name': 'pred-tracks'
, 'nontrigger_input_dir': '/ssd5/giorgian/tracks-pileup-october-2024-trigger/nontrigger/0/', 'rescale_by_percentile': -1, 'trigger_input_dir': '/ssd5/giorgian/tracks-pileup-october-2024-trigger/trigger/1/', 'use_center': True, 'use_energy': False, 'use_geometric_features': True, 'use_momentum': False, 'use_n_hits': True, 'use_n_pixels': False, 'use_parallel_momentum': False, 'use_predicted_pz': True, 'use_radius': True, 'use_transverse_momentum': False}

os.makedirs(TRIGGER_OUTPUT_DIR, exist_ok=True)
os.makedirs(NONTRIGGER_OUTPUT_DIR, exist_ok=True)


train_data, val_data, test_data = get_data_loaders(**data_config)

dataset = train_data.dataset.dataset

def process(i):
    ev = dataset[i]
    out_filename = os.path.basename(dataset.filenames[i]).replace('.npz', '.txt')
    if ev.trigger:
        out_file = os.path.join(TRIGGER_OUTPUT_DIR, out_filename)
    else:
        out_file = os.path.join(NONTRIGGER_OUTPUT_DIR, out_filename)


    tracks = ev.track_vector[:, :15]
    geo_features = ev.track_vector[:, 15:15+13]
    radii = ev.track_vector[:, 15+13]
    centers = ev.track_vector[:, 15+13+1:15+13+1+2]
    p_z = ev.track_vector[:, 15+13+1+1+2]
    n_hits = ev.track_vector[:, 15+13+1+2+1:]
    with open(out_file, 'w') as fout:
        print(f'Here is a particle collision event with {len(tracks)} tracks.', file=fout)
                    
        for i, ti in enumerate(np.random.permutation(tracks.shape[0])):
                print(f'Track number {i+1} has a transverse momentum of {radii[ti]}, a parallel momentum of {p_z[ti]}, a center of {tuple(centers[ti].tolist())} and a trajectory of {tuple(tracks[ti].tolist())} as the particle flew through the detector.', file=fout)


Parallel(n_jobs=16)(delayed(process)(i) for i in tqdm.tqdm(range(len(dataset))))
