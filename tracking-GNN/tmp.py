import numpy as np
from joblib import Parallel, delayed
import glob
from numpy import dtype
from tqdm import tqdm

FILE_DIR_PATH = '/ssd3/giorgian/hits-data-mixed/trigger/1'

def fix(filename):
    with np.load(filename, allow_pickle=True) as data:
        data_dict = {k: data[k] for k in data.keys()}
        if data_dict['track_origin'].dtype == dtype('O'):
            data_dict['track_origin'] = np.stack(data_dict['track_origin'])
            np.savez(filename, **data_dict)

files = glob.glob(f'{FILE_DIR_PATH}/*.npz')
Parallel(n_jobs=16)(delayed(fix)(filename) for filename in tqdm(files))
