import numpy as np
import glob
import os
import tqdm
from joblib import Parallel, delayed

TRIGGER_DIR = '/ssd2/giorgian/hits-data-august-2022/trigger/1/'
NONTRIGGER_DIR = '/ssd2/giorgian/hits-data-august-2022/nontrigger/0/'

files = glob.glob(os.path.join(TRIGGER_DIR, '*.npz')) + glob.glob(os.path.join(NONTRIGGER_DIR, '*.npz'))

def process(file):
    with np.load(file) as f:
        p = f['momentum']
        energy = f['energy']
        n_hits = f['hit_cartesian'].shape[0]
        p_t = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        p_t = p_t[~np.isnan(p_t)]
        energy = energy[~np.isnan(energy)]
        trigger = f['trigger']
        return np.max(p_t, initial=0), np.max(energy, initial=0), n_hits, trigger

data = Parallel(n_jobs=16)(delayed(process)(file) for file in tqdm.tqdm(files))
data = np.array(data)
np.savez('trigger_data.npz', p_t=data[:, 0], energy=data[:, 1], n_hits=data[:, 2], trigger=data[:, 3])
