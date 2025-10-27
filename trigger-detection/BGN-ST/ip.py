import numpy as np
import tqdm
import glob
DIR_1 = '/ssd2/giorgian/HFML-NewFiles-1/trigger/1'
DIR_2 = '/ssd2/giorgian/HFML-NewFiles-1/nontrigger/0'

# get the files in each directory
files_1 = glob.glob(DIR_1 + '/*.npz')
files_2 = glob.glob(DIR_2 + '/*.npz')
# combine the files
files = files_1 + files_2
# shuffle the files
np.random.shuffle(files)

ips = []
for file in tqdm.tqdm(files[:1000000]):
    ip = np.load(file)['ip']
    ips.append(ip)

ips = np.stack(ips, axis=0)
# Calculate the mean and std
mean = np.mean(ips, axis=0)
std = np.std(ips, axis=0)
print('mean:', mean)
print('std:', std)
