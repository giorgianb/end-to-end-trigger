import numpy as np
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

files = glob.glob('/secondssd/giorgian/hits-data-october-2024/trigger/1/*.npz')

def fix(file):
    f = np.load(file, allow_pickle=True)
    f_dict = {k:f[k] for k in f.keys()}
    f_dict['track_origin'] = np.stack(f['track_origin'], axis=0)
    np.savez(file, **f_dict)

Parallel(n_jobs=10)(delayed(fix)(file) for file in tqdm(files))

