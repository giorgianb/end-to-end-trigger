#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dataclasses import replace
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import os.path
import sys
import logging
import pickle
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from models.garnet_trigger import GNNGraphClassifier
from models.mlp_trigger_layerwise import GNNGraphClassifier
from icecream import ic
from numpy.linalg import inv
import sklearn.metrics as metrics
from datasets import get_data_loaders
from tqdm import tqdm


# In[2]:


DEVICE = "cuda:0"


# In[3]:


# create model and load checkpoint
model_result_folder = '/home1/giorgian/projects/trigger-detection-pipeline/sPHENIX/trigger_results/agnn/agnn-lr8.19806576478371e-05-b12-d71-PReLU-gi1-ln-True-n1600000/experiment_2024-04-26_13:41:37'
model_result_folder = '/home1/giorgian/projects/trigger-detection-pipeline/sPHENIX/trigger_results/agnn/agnn-lr9.99230738912354e-05-b12-d214-ReLU-gi1-ln-True-n1600000/experiment_2024-05-02_21:13:56'

config_file = model_result_folder + '/config.pkl'
config = pickle.load(open(config_file, 'rb'))
data_config = config.get('data')
dphi_max, dz_max = data_config['phi_slope_max'], data_config['z0_max']

model_config = config.get('model', {})
model_config.pop('loss_func')
print(f'{model_config["name"]=}')
model_config.pop('name')
model = GNNGraphClassifier(**model_config).to(DEVICE)
print(f'{model_config=}')

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


print(f'{data_config["input_dir2"]=}')


# In[5]:


print(f'{data_config=}')


# In[6]:


data_config['n_train'] = 1
data_config['n_valid'] = 994440-1
#data_config['input_dir2'] = '/ssd1/giorgian/hits-data-august-2022-ctypes/trigger/1'
#data_config['force_inputdir2_nontrigger'] = True


# In[ ]:





# In[7]:


train_data_loader, valid_data_loader = get_data_loaders(distributed=False, rank=0, n_ranks=0, **data_config)


# In[ ]:


preds, targets = [], []
sigmoid = torch.nn.Sigmoid()
model.eval()
for batch in tqdm(valid_data_loader):
    preds.extend(model(batch.to(DEVICE)).squeeze(0).detach().cpu().numpy())
    targets.extend(batch.trigger.squeeze(0).detach().cpu().numpy())

preds = np.array(preds)
targets = np.array(targets)
np.savez('mlp_trigger_layeriwse_mvtx.npz', preds=preds, targets=targets, model_result_folder=model_result_folder)


# In[ ]:


def efficiency_purity(correct, preds_prob, signal_mix=0.01):
    signal = np.where(correct)[0]
    background = np.where(1 - correct)[0]
    mix = len(signal)/len(correct)
    drop = int(np.ceil((len(signal) - signal_mix * len(correct))/(1 - signal_mix)))
    np.random.shuffle(signal)
    keep = signal[:len(signal) - drop]
    keep = np.concatenate([keep, background], axis=0)
    c = correct[keep]
    p = preds_prob[keep]
    # Calculate efficiency
    tp = np.sum(c * (p > 0.5))
    tn = np.sum(( 1- c) * (p <= 0.5))
    fp = np.sum((1 - c) * (p > 0.5))
    fn = np.sum(c * (p <= 0.5))

    # effiency is how much of the signal we captured
    efficiency = tp/(tp + fn)
    # purity is how much of the signal is true
    purity = tp/(tp + fp)
    brr = tn/(tn + fp)

    return efficiency, purity, brr

def efficiency_purity_2(correct, preds_prob, signal_mix=0.01):
    signal = np.where(correct)[0]
    background = np.where(1 - correct)[0]
    mix = len(signal)/len(correct)
    drop = int(np.ceil((len(signal) - signal_mix * len(correct))/(1 - signal_mix)))
    np.random.shuffle(signal)
    keep = signal[:len(signal) - drop]
    keep = np.concatenate([keep, background], axis=0)
    c = correct[keep]
    print(f'{np.sum(c)/c.shape[0]=}')
    p = preds_prob[keep]
    # Calculate efficiency
    cutoffs = np.unique(np.sort(preds_prob))
    if len(cutoffs) > 10000:
        factor = int(len(cutoffs)/10000)
        cutoffs = cutoffs[::factor]
    efficiencies = []
    purities = []
    brrs = []
    for cutoff in cutoffs:
        tp = np.sum(c * (p > cutoff))
        tn = np.sum(( 1- c) * (p <= cutoff))
        fp = np.sum((1 - c) * (p > cutoff))
        fn = np.sum(c * (p <= cutoff))

        # effiency is how much of the signal we captured
        efficiency = tp/(tp + fn) if tp + fn != 0 else 0
        # purity is how much of the signal is true
        purity = tp/(tp + fp) if tp + fp != 0 else 0
        brr = tn / (tn + fp) if tn + fp != 0 else 0

        efficiencies.append(efficiency)
        purities.append(purity)
        brrs.append(brr)
        
    return np.array(efficiencies), np.array(purities), np.array(brrs)


# In[12]:


brrs


# In[13]:


print(brrs)


# In[1]:


correct.shape


# In[79]:


t = np.unique(np.sort(preds))


# In[80]:


t.shape


# In[28]:


preds_prob = np.array(preds)
targets = np.array(targets)


# In[29]:


targets.shape


# In[30]:


accs_gt = np.sum((preds_prob > 0) == targets, axis=-1)/targets.shape[0]
accs_gt


# In[13]:


preds_prob.shape


# In[14]:


targets.shape


# In[15]:


efficiency_purity(targets, preds_prob, signal_mix=0.01)


# In[16]:


efficiency_purity(targets, preds_prob, signal_mix=0.001)


# In[15]:


np.sum(targets == 0)


# In[18]:


efficiencies, purities, brrs = efficiency_purity_2(targets, preds_prob, signal_mix=0.01)


# In[31]:


for brr in [0.9, 0.95, 0.99, 0.999]:
    i = np.argmin(np.abs(brrs - brr))
    print(f'Efficiency: {efficiencies[i]*100:.4f}, Purity: {purities[i]*100:.4f}% BRR: {brrs[i]*100:.4f}%')


# In[33]:


efficiencies.shape


# In[18]:


efficiencies_1, purities_1, brrs_1 = efficiency_purity_2(targets, preds_prob, signal_mix=0.001)


# In[19]:


for brr in [0.9, 0.95, 0.99, 0.999]:
    i = np.argmin(np.abs(brrs_1 - brr))
    print(f'Efficiency: {efficiencies_1[i]*100:.4f}, Purity: {purities_1[i]*100:.4f}% BRR: {brrs_1[i]*100:.4f}%')


# In[24]:


accs_gt


# In[69]:


i = np.argmin(np.abs(brrs-0.99))


# In[71]:





# In[70]:


efficiencies[i]*100, purities[i]*100


# In[24]:


i = np.argmin(np.abs(brrs_1-0.99))


# In[25]:


efficiencies_1[i]*100, purities_1[i]*100


# In[20]:


plt.style.use('ggplot')
plt.plot(efficiencies, purities)
plt.xlabel('Efficiency')
plt.ylabel('Purity')
plt.title('Ground-Truth Track Efficiency/Purity Plot')


# In[21]:


plt.style.use('ggplot')
plt.plot(efficiencies, 1/(1 - brrs))
plt.xlabel('Efficiency')
plt.ylabel('$\\frac{1}{1 - BRR}$')
plt.title('Ground-Truth Track Efficiency/BRR Plot (1% Signal Mix)')
for brr in [0.9, 0.95, 0.99, 0.999]:
    i = np.argmin(np.abs(brrs - brr))
    x, y, p = efficiencies[i], brrs[i], purities[i]
    plt.axvline(x, color='black')
    plt.text(x+0.02, 2, f'E={x*100:.2f}%\nB={y*100:.2f}%\nP={p*100:.2f}%', fontsize=6)
    #plt.text(x, 0.1, f'=({x:.2f}, {y:.2f})', ha='right', va='bottom', fontsize=7)
    #plt.text(x, y, f'({x:.2f}, {y:.2f})', ha='right', va='bottom')
    print(f'{efficiencies[i]*100:.2f}%\t{purities[i]*100:.2f}%\t{brrs[i]*100:.2f}%')
plt.yscale('log')


# In[22]:


plt.style.use('ggplot')
plt.plot(efficiencies_1, 1/(1 - brrs_1))
plt.xlabel('Efficiency')
plt.ylabel('Background Rejection Rate')
plt.title('Ground-Truth Track Efficiency/BRR Plot (0.1% Signal Mix)')
for brr in [0.9, 0.95, 0.99, 0.999]:
    i = np.argmin(np.abs(brrs - brr))
    x, y, p = efficiencies_1[i], brrs_1[i], purities_1[i]
    plt.axvline(x, color='black')
    plt.text(x+0.02, 2, f'E={x*100:.2f}%\nB={y*100:.2f}%\nP={p*100:.2f}%', fontsize=6)
    #plt.text(x, 0.1, f'=({x:.2f}, {y:.2f})', ha='right', va='bottom', fontsize=7)
    #plt.text(x, y, f'({x:.2f}, {y:.2f})', ha='right', va='bottom')
    print(f'{efficiencies[i]*100:.2f}%\t{purities[i]*100:.2f}%\t{brrs[i]*100:.2f}%')
    
plt.yscale('log')


# In[60]:


purities


# In[26]:


accs_pred_phi, accs_gt_phi


# In[ ]:


plt.plot(np.linspace(0, 2*np.pi, 40), accs_gt)
x_ticks = np.arange(0, (2 + 1/2)*np.pi, np.pi/2)
print(x_ticks)
ax = plt.gca()  # Get the current axis
ax.set_xticks(x_ticks)
ax.set_xticklabels(["0", "$\\frac{π}{2}$", "π", "$\\frac{3π}{2}$", "2π"])
plt.xlabel('Rotation Angle Applied')
plt.ylabel('Accuracy')
plt.title("Effects of Rotation on Accuracy (GT Tracks)")


# In[43]:


plt.style.use('ggplot')

plt.plot(np.linspace(0, 2*np.pi, 40), accs_gt_phi, label='Ground Truth Tracks')
plt.plot(np.linspace(0, 2*np.pi, 40), accs_pred_phi, label='Predicted Tracks')
x_ticks = np.arange(0, (2 + 1/2)*np.pi, np.pi/2)
ax = plt.gca()  # Get the current axis
ax.set_xticks(x_ticks)
ax.set_xticklabels(["0", "$\\frac{π}{2}$", "π", "$\\frac{3π}{2}$", "2π"])
plt.xlabel('Rotation Angle Applied')
plt.ylabel('Accuracy')
plt.title("Effects of Rotation on Accuracy")
plt.ylim(0.84, 0.92)
plt.legend()


# In[11]:


N_BATCHES = 1
all_preds = []
all_preds_prob = []
all_correct = []
val_data.dataset.dataset.phi = 0
for z in np.linspace(-20, 20, 10):
    preds = []
    preds_prob = []
    correct = []
    val_data.dataset.dataset.z = z
    for batch in tqdm(islice(val_data, 0, N_BATCHES), total=N_BATCHES):
        mask = torch.zeros(batch.track_vector.shape[:-1])
        for i, n in enumerate(batch.n_tracks):
            mask[i, :n] = 1
            

        mask = mask.to(DEVICE)
        track_vector = batch.track_vector.to(DEVICE)
        n_batches, n_tracks = track_vector.shape[:2]
        hits = track_vector[:, :, :15].reshape(n_batches, n_tracks, 5, 3)
        
        
        is_trigger_track = batch.is_trigger_track.to(DEVICE, torch.bool)
        trigger = (batch.trigger.to(DEVICE) == 1).unsqueeze(-1)

        mask_logits = model(track_vector, mask)
        pred = mask_logits.max(dim=1)[1]
        preds.extend(pred.cpu().data.numpy())
        preds_prob.extend(nn.Softmax(dim=1)(mask_logits)[:, 1].detach().cpu().numpy().flatten())
        correct.extend(trigger.detach().cpu().numpy().flatten())
    all_correct.append(correct)
    all_preds.append(preds)
    all_preds_prob.append(preds_prob)


# In[14]:


plt.plot(np.linspace(-20, 20, 10), accs_pred_z)
plt.xlabel('Z Translation Applied')
plt.ylabel('Accuracy')
plt.title("Effects of Z Translation on Accuracy (GT Tracks)")


# In[13]:


all_preds = np.array(all_preds)
all_correct = np.array(all_correct)
accs_pred_z = np.sum(all_preds == all_correct, axis=-1)/all_correct.shape[1]


# In[37]:


plt.plot(np.linspace(-20, 20, 40), accs_gt)
plt.xlabel('Z Translation Applied')
plt.ylabel('Accuracy')
plt.title("Effects of Z Translation on Accuracy (GT Tracks)")


# In[52]:


plt.style.use('ggplot')
plt.plot(np.linspace(-20, 20, 40), accs_gt_z, label='Ground Truth Tracks')
plt.plot(np.linspace(-20, 20, 40), accs_pred_z, label='Predicted Tracks')
plt.xlabel('Z Translation Applied')
plt.ylabel('Accuracy')
plt.title("Effects of Z Translation on Accuracy")
plt.legend()


# In[34]:


accs_gt_z, accs_pred_z


# # Collect Statistics on Z-Distribution

# In[47]:


N_BATCHES = 400
zs = []
val_data.dataset.dataset.phi = 0
val_data.dataset.dataset.z = 0


for batch in tqdm(islice(val_data, 0, N_BATCHES), total=N_BATCHES):
    mask = torch.zeros(batch.track_vector.shape[:-1])
    for i, n in enumerate(batch.n_tracks):
        mask[i, :n] = 1

    mask = mask.to(DEVICE)
    track_vector = batch.track_vector.to(DEVICE)
    track_vector = track_vector * mask.unsqueeze(-1)
    n_batches, n_tracks = track_vector.shape[:2]
    hits = track_vector[:, :, :15].reshape(n_batches, n_tracks, 5, 3)
    good_hits = torch.any(hits != 0, dim=-1)
    zs.extend(hits[..., -1][good_hits].cpu().detach().numpy())


# In[39]:


gt_zs = np.array(zs)


# In[48]:


pred_zs = np.array(zs)


# In[54]:


plt.style.use('ggplot')
plt.hist(pred_zs, bins=30, alpha=0.5, label='Predicted Tracks')
plt.hist(gt_zs, bins=30, alpha=0.5, label='Ground Truth Tracks')
plt.xlabel('Z-Coordinate')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Z-Coordinate')


# In[49]:


help(plt.hist)


# # Track Dropping

# In[67]:


N_BATCHES = 80
# No dropping
all_preds = []
all_preds_prob = []
# Drop TT
tt_preds = []
tt_preds_prob = []
# Drop NT
nt_preds = []
nt_preds_prob = []

correct = []
val_data.dataset.dataset.phi = 0
val_data.dataset.dataset.z = 0

for batch in tqdm(islice(val_data, 0, N_BATCHES), total=N_BATCHES):
    mask = torch.zeros(batch.track_vector.shape[:-1])
    nt_mask = torch.zeros(batch.track_vector.shape[:-1])
    for i, n in enumerate(batch.n_tracks):
        mask[i, :n] = 1
        nt_mask[i, :n] = 1
        n_trigger_tracks = torch.sum(batch.is_trigger_track[i].to(torch.bool))
        non_trigger_tracks = torch.where(~batch.is_trigger_track[i, :n].to(torch.bool))[0].numpy().tolist()
        #print(f'{non_trigger_tracks=}')
        random.shuffle(non_trigger_tracks)
        #print(f'{non_trigger_tracks=}')
        # We are randomly dropping the same amount of non-trigger tracks as of trigger tracks
        for j in non_trigger_tracks[:n_trigger_tracks]:
            #print(f'Setting {j} to 0')
            nt_mask[i, j] = 0
        #if n_trigger_tracks != 0:
           # print(f'{n_trigger_tracks=}\n{mask[i, :n]=}\n{nt_mask[i, :n]=}')
        

    mask = mask.to(DEVICE)
    nt_mask = nt_mask.to(DEVICE)
    track_vector = batch.track_vector.to(DEVICE)

    is_trigger_track = batch.is_trigger_track.to(DEVICE, torch.bool)
    trigger = (batch.trigger.to(DEVICE) == 1).unsqueeze(-1)

    mask_logits = model(track_vector, mask)
    pred = mask_logits.max(dim=1)[1]
    all_preds.extend(pred[trigger.flatten()].cpu().numpy())
    all_preds_prob.extend(nn.Softmax(dim=1)(mask_logits[trigger.flatten()])[:, 1].detach().cpu().numpy().flatten())
    # Drop the trigger-tracks
    tt_mask = mask * ~is_trigger_track
    mask_logits = model(track_vector, tt_mask)
    tt_preds.extend(pred[trigger.flatten()].cpu().numpy())
    tt_preds_prob.extend(nn.Softmax(dim=1)(mask_logits[trigger.flatten()])[:, 1].detach().cpu().numpy().flatten())

    mask_logits = model(track_vector, nt_mask)
    nt_preds.extend(pred[trigger.flatten()].cpu().numpy())
    nt_preds_prob.extend(nn.Softmax(dim=1)(mask_logits[trigger.flatten()])[:, 1].detach().cpu().numpy().flatten())

    
    correct.extend(trigger.detach().cpu().numpy().flatten())



# In[58]:


all_pred = np.array(all_preds_prob)
all_pred_tt = np.array(tt_preds_prob)
all_pred_nt = np.array(nt_preds_prob)


# In[68]:


all_gt = np.array(all_preds_prob)
all_gt_tt = np.array(tt_preds_prob)
all_gt_nt = np.array(nt_preds_prob)


# In[59]:


len(all_pred)


# In[73]:


plt.style.use('ggplot')
plt.boxplot([all_pred, all_pred_nt, all_pred_tt, all_gt, all_gt_tt, all_gt_nt])
plt.xticks([1, 2, 3, 4, 5, 6], ['P AT', 'P TT + SNT', 'P NT', 'GT AT', 'GT TT + NT', 'GT SNT'])  # Adjust or add more labels based on your datasets
plt.ylabel('Trigger Probability')
plt.title('Effect of Track Dropping on Trigger Probability Distribution')


# # Hit-Dropping

# In[76]:


N_BATCHES = 80
# No dropping
pred_probs = [list() for i in range(5)]


val_data.dataset.dataset.phi = 0
val_data.dataset.dataset.z = 0

for layer_drop in range(5):
    for batch in tqdm(islice(val_data, 0, N_BATCHES), total=N_BATCHES):
        mask = torch.zeros(batch.track_vector.shape[:-1])
        for i, n in enumerate(batch.n_tracks):
            mask[i, :n] = 1


        mask = mask.to(DEVICE)
        track_vector = batch.track_vector.to(DEVICE)
        n_batches, n_tracks = track_vector.shape[:2]
        hits = track_vector[..., :15].reshape(n_batches, n_tracks, 5, 3)
        hits[:, :, layer_drop, :] = 0
        track_vector[..., :15] = hits.reshape(n_batches, n_tracks, 15)

        is_trigger_track = batch.is_trigger_track.to(DEVICE, torch.bool)
        trigger = (batch.trigger.to(DEVICE) == 1).unsqueeze(-1)

        mask_logits = model(track_vector, mask)
        pred = mask_logits.max(dim=1)[1]
        pred_probs[layer_drop].extend(nn.Softmax(dim=1)(mask_logits[trigger.flatten()])[:, 1].detach().cpu().numpy().flatten())
    


# In[77]:


plt.boxplot(pred_probs)


# # Noise Std

# In[27]:


N_BATCHES = 40
all_preds = []
all_preds_prob = []
all_correct = []
val_data.dataset.dataset.phi = val_data.dataset.dataset.z = 0
noise_stds = np.linspace(-6, -1, 30)
#noise_stds = [-np.inf]
for noise_std in noise_stds:
    val_data.dataset.dataset.noise_std = 10**noise_std
    preds = []
    preds_prob = []
    correct = []
    for batch in tqdm(islice(val_data, 0, N_BATCHES), total=N_BATCHES):
        mask = torch.zeros(batch.track_vector.shape[:-1])
        for i, n in enumerate(batch.n_tracks):
            mask[i, :n] = 1
            

        mask = mask.to(DEVICE)
        track_vector = batch.track_vector.to(DEVICE)
        n_batches, n_tracks = track_vector.shape[:2]
        hits = track_vector[:, :, :15].reshape(n_batches, n_tracks, 5, 3)
        
        
        is_trigger_track = batch.is_trigger_track.to(DEVICE, torch.bool)
        trigger = (batch.trigger.to(DEVICE) == 1).unsqueeze(-1)

        mask_logits = model(track_vector, mask)
        pred = mask_logits.max(dim=1)[1]
        preds.extend(pred.cpu().data.numpy())
        preds_prob.extend(nn.Softmax(dim=1)(mask_logits)[:, 1].detach().cpu().numpy().flatten())
        correct.extend(trigger.detach().cpu().numpy().flatten())
    all_correct.append(correct)
    all_preds.append(preds)
    all_preds_prob.append(preds_prob)


# In[28]:


all_preds = np.array(all_preds)
all_correct = np.array(all_correct)
accs_gt_noise = np.sum(all_preds == all_correct, axis=-1)/all_correct.shape[1]


# In[22]:


accs_gt_noise


# In[15]:


accs_pred_noise


# In[23]:


accs_gt_noise


# In[12]:


accs_gt_noise, accs_pred_noise


# In[20]:


accs_gt_noise


# In[18]:


accs_pred_noise


# In[29]:


plt.style.use('ggplot')
plt.plot(10**noise_stds, accs_gt_noise, label='Ground Truth Tracks')
plt.plot(10**noise_stds, accs_pred_noise, label='Predicted Tracks')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.title("Effects of Noise on Accuracy")
plt.legend()


# In[ ]:




