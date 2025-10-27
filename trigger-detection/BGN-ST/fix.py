import numpy as np
from joblib import Parallel, delayed
import os.path
import tqdm
import glob
import random

TRIGGER_INPUT_DIR = '/ssd2/giorgian/real-tracks-data-mixed-2/trigger/1/'
NONTRIGGER_INPUT_DIR = '/ssd2/giorgian/real-tracks-data-mixed-2/nontrigger/1/'
files = list(glob.glob(os.path.join(TRIGGER_INPUT_DIR, '*.npz'))) + list(glob.glob(os.path.join(NONTRIGGER_INPUT_DIR, '*.npz')))


def fix_file(file):
    try:
        data = np.load(file)
    except Exception as e:
        print(f'Failed to load {file}: {e}')

Parallel(n_jobs=16)(delayed(fix_file)(file) for file in tqdm.tqdm(files))

