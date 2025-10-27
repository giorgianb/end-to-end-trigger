#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
import pickle
import dataloaders
import matplotlib.pyplot as plt
from utils.log import load_checkpoint
from collections import defaultdict
from itertools import islice
from tqdm import tqdm
from models.Bipartite_Attention_Masked import Bipartite_Attention as Model
import matplotlib as mpl
import torch.nn as nn
import numpy as np
import random


# In[3]:


from importlib import reload


# In[4]:


config_file_path = 'train_results/biatt-blri/experiment_2024-10-17_13:41:02/config.pkl'
config_file_path = 'train_results/biatt-augment-adj/experiment_2024-10-23_22:12:29/config.pkl'
#config_file_path = '/disks/disk1/giorgian/old-trainings/biatt-augment-adj/experiment_2023-08-03_21:42:12/config.pkl'
with open(config_file_path, 'rb') as f:
    config = pickle.load(f)


# In[5]:


dconfig = config['data']
dconfig['n_train'] = 1
dconfig['n_valid'] = 317104 - 2
dconfig['n_test'] = 1
dconfig['trigger_input_dir'] =  '/ssd2/giorgian/real-tracks-data-mixed-3/trigger/1/'
dconfig['nontrigger_input_dir'] =  None
dconfig['batch_size'] = 1
dconfig['n_workers'] = 0

train_data, val_data, test_data = dataloaders.get_data_loaders(**dconfig)


# In[6]:


DEVICE = 'cuda:0'
# α=0.7
checkpoint_file = 'train_results/biatt-augment-adj/experiment_2024-10-23_22:12:29/checkpoints/model_checkpoint_014.pth.tar'

#checkpoint_file = '/disks/disk1/giorgian/old-trainings/biatt-augment-adj/experiment_2023-08-03_21:42:12/checkpoints/model_checkpoint_016.pth.tar'
mconfig = config['model']
model = Model(**mconfig)
model = load_checkpoint(checkpoint_file, model)
model = model.to(DEVICE)
model.eval()


# In[ ]:


preds = []
track_vectors = []

i = 0
model.eval()
for batch, mask in tqdm(val_data):
    tracks = batch.track_vector.to(DEVICE)
    if tracks.shape[1] == 0:
        continue
        
    mask = mask.to(DEVICE)
    preds.append(model(tracks, mask)[:, -1].detach().cpu().numpy())
    track_vectors.append(batch.track_vector.cpu().numpy())
    i += 1

    del tracks
    del batch
    del mask


    if i %  100 == 0:
        save = {'preds': preds, 'track_vectors': track_vectors}
        with open('/disks/disk1/giorgian/preds_4.pkl', 'wb') as f:
            pickle.dump(save, f)
