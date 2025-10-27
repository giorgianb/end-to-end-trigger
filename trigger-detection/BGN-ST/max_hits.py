import numpy as np
from joblib import Parallel, delayed
import os.path
import tqdm
import glob
import random

TRIGGER_INPUT_DIR = '/ssd1/giorgian/tracks-data-august-2022/trigger/1'
NONTRIGGER_INPUT_DIR = '/ssd1/giorgian/tracks-data-august-2022/nontrigger/0'
TRIGGER_OUTPUT_DIR = '/ssd1/giorgian/tracks-data-august-2022-fixed/trigger/1'
NONTRIGGER_OUTPUT_DIR = '/ssd1/giorgian/tracks-data-august-2022-fixed/nontrigger/0'
files = list(glob.glob(os.path.join(TRIGGER_INPUT_DIR, '*.npz'))) + list(glob.glob(os.path.join(NONTRIGGER_INPUT_DIR, '*.npz')))
print(f'we have {len(files)} files')
random.seed(42)
random.shuffle(files)

# make output dirs
os.makedirs(TRIGGER_OUTPUT_DIR, exist_ok=True)
os.makedirs(NONTRIGGER_OUTPUT_DIR, exist_ok=True)

def analyze_file(file):
    try:
        data = np.load(file)
        return np.max(np.sum(f['hit_type'] != -1, axis=-1))
    except Exception as e:
        print(f'Failed to fix {file}: {e}')
        return 0

max_n_hits = Parallel(n_jobs=16)(delayed(fix_file)(file) for file in tqdm.tqdm(files))
