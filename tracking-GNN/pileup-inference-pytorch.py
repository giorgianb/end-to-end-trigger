from dataclasses import replace
import numpy as np
import os
import torch
import os.path
import sys
import logging
import tqdm
import pickle
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from models.agnn_inference import GNNSegmentClassifier
from icecream import ic
from numpy.linalg import inv
import sklearn.metrics as metrics
from datasets import get_data_loaders
from datasets.hit_graph_trigger_pileup import load_file
import dataclasses

is_check_acc = True
is_save = True
DEVICE = 'cuda:1'
start_file = 0
end_file = int(1e6)

# server 16
trigger_input_dir = "/ssd3/giorgian/hits-data-october-2024/trigger/1/"
nontrigger_input_dir = "/ssd2/giorgian/hits-data-august-2022/nontrigger/0/"
if is_save:
    # old_output_dir = '/home/tingtingxuan/Data/7layer_data/tracking_inference_INTTclustered_postrack/nontrigger'
    #trigger_output_dir = '/disks/disk1/giorgian/tracks-pileup-october-2024-trigger/trigger/1/'
    #nontrigger_output_dir = '/disks/disk1/giorgian/tracks-pileup-october-2024-trigger/nontrigger/0/'
    trigger_output_dir = '/ssd3/giorgian/shuyang-hits-2024/trigger/1/'
    nontrigger_output_dir = '/ssd3/giorgian/shuyang-hits-2024/nontrigger/0/'
    os.makedirs(trigger_output_dir, exist_ok=True)
    os.makedirs(nontrigger_output_dir, exist_ok=True)

# model_result_folder = '/home1/tingtingxuan/trigger-detection-pipeline/tracking_result/tracking/tracking-gnn-7layer-highmtrack-lr0.0001-b32-d1024-Tanh-gi8-ln-True-n200000/experiment_2022-10-01_18:34:01'
#model_result_folder = '/home1/tingtingxuan/sPHENIX/tracking-GNN/tracking_result/tracking/tracking-gnn-7layer-alltrack-fix-lr0.0001-b32-d1024-Tanh-gi8-ln-True-n200000/experiment_2022-10-08_17:08:01'
#model_result_folder = '/home1/giorgian/projects/trigger-detection-pipeline/sPHENIX/tracking_results/agnn/agnn-lr0.006720478856559648-b512-d8-ReLU-gi1-ln-False-n1000000/experiment_2023-07-11_06:43:41/'
#model_result_folder = 'tracking_results/train/tracking-gnn-lr0.0001-b12-d64-ReLU-gi2-ln-True-n10000/experiment_2024-10-04_10:14:52/'
#model_result_folder = 'tracking_results/train/tracking-gnn-lr0.0001-b12-d64-ReLU-gi2-ln-True-n100000/experiment_2024-10-10_10:00:43'
#model_result_folder = '../tracking_results/agnn/agnn-lr0.003039158491105233-b12-d64-PReLU-gi3-ln-True-n40000/experiment_2024-11-09_23:30:24'
#model_result_folder = '../tracking_results/agnn/agnn-lr0.006488739557034936-b64-d64-PReLU-gi3-ln-True-n80000/experiment_2024-11-22_04:07:25/'
#model_result_folder = '../tracking_results/agnn/agnn-lr0.006488739557034936-b64-d64-PReLU-gi3-ln-True-n80000/experiment_2024-11-22_04:07:25/'
model_result_folder = '../tracking_results/agnn/agnn-lr0.004229328554496046-b64-d64-ReLU-gi1-ln-True-n80000/experiment_2024-11-26_19:53:03/'







trigger_filenames = sorted([os.path.join(trigger_input_dir, f) for f in os.listdir(trigger_input_dir) if f.startswith('event')])
nontrigger_filenames = sorted([os.path.join(nontrigger_input_dir, f) for f in os.listdir(nontrigger_input_dir) if f.startswith('event')])
# filenames = np.random.choice(filenames, file_num)
#print(f'Input Dir: {input_dir}')
#print(f'Tracking accuracy check for {len(filenames)} files.')


# create model and load checkpoint
config_file = model_result_folder + '/config.pkl'
config = pickle.load(open(config_file, 'rb'))
data_config = config.get('data')
print(f'{data_config=}')
dphi_max, dz_max = data_config['phi_slope_max'], data_config['z0_max']

model_config = config.get('model', {})
model_config.pop('loss_func')
model_config.pop('name')
model = GNNSegmentClassifier(**model_config).to(DEVICE)

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

sigmoid = torch.nn.Sigmoid()

if is_check_acc:
    preds = []
    labels = []


BATCH_SIZE = 64

exists = 0
dconfig = data_config
dconfig['batch_size'] = BATCH_SIZE
dconfig['load_all'] = True
dconfig['load_full_event'] = True
dconfig['n_workers'] = 16
#dconfig['ramp_up_nmix'] = False
#dconfig['n_mix'] = 20
#dconfig['input_dir'] = '/ssd3/giorgian/hits-data-mixed/trigger/1'
#dconfig['input_dir2'] = '/ssd3/giorgian/hits-data-mixed/nontrigger/1'
#dconfig['input_dir'] = trigger_input_dir
#dconfig['input_dir2'] = nontrigger_input_dir
#print(f'{dconfig["n_mix"]=}')
#print(f'{dconfig=}')

#dconfig['n_train'] = len(trigger_filenames) + len(nontrigger_filenames)
#dconfig['n_valid'] = 0

train_data_loader, valid_data_loader = get_data_loaders(distributed=False, rank=0, n_ranks=0, **dconfig)
for i, batch in enumerate(tqdm.tqdm(train_data_loader, smoothing=0.0)):
    batch = batch.to(DEVICE)

    batch_output = model((batch.x, batch.edge_index))
    e = sigmoid(model((batch.x, batch.edge_index))).detach().cpu().numpy().reshape(-1, 1)
    #data = dataclasses.asdict(load_file(filename))
    edge_index = batch.edge_index.cpu().numpy()
    x = batch.x.cpu().numpy()

    if is_check_acc:

        label = batch.y.cpu().numpy().reshape(-1)
        
        labels.append(label)
        preds.append(e.reshape(-1)>0.5)
    
    if is_save:
        b = batch.batch.cpu().numpy()
        for i in range(BATCH_SIZE):
            filename = batch.filename[i]
            base = os.path.basename(filename)
            e = e.reshape(-1)
            fn = os.path.basename(filename)
            
            model_edge_index = edge_index[:, b[edge_index[0]] == i]
            model_edge_probability = e[b[edge_index[0]] == i]
            data = dataclasses.asdict(batch.event_info[i])
            data['x'] = x[b == i]

            #print(f'{data["hit_cartesian"].shape=} {model_edge_index.shape=} {model_edge_probability.shape=} {data["edge_index"].shape=} {data["trigger"]}=')
            if data['trigger']:
                out_filename = os.path.join(trigger_output_dir, base)
                np.savez(out_filename, **data, model_edge_probability=model_edge_probability, model_edge_index=model_edge_index)
            else:
                out_filename = os.path.join(nontrigger_output_dir, base)
                np.savez(out_filename, **data, model_edge_probability=model_edge_probability, model_edge_index=model_edge_index)



if is_check_acc:
    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {'prec': metrics.precision_score(labels, preds>0),
                'recall': metrics.recall_score(labels, preds>0),
                'acc': metrics.accuracy_score(labels, preds>0),
                'F1': metrics.f1_score(labels, preds>0)}

    print(result)
