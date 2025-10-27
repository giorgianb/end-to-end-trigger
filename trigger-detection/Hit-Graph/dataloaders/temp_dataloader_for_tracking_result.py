# from collections import namedtuple
# # System imports
# import os
# import random

# # External imports
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, random_split, Sampler
# import torch_geometric
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data.dataloader import default_collate
# def load_file(filename):

#     with np.load(filename) as f:
#         trigger_flag = f['trig']
#         edge_index = f['edge_index']
#         hits = f['hits_info']
#         # hits = np.concatenate([hits, np.reshape([hits.shape[0]] * hits.shape[0], (-1, 1))], 1)
#         e = f['e']
#         n_hits = f['n_hits']
#         n_edges = f['n_edges']

#     return hits, edge_index, trigger_flag, e, n_hits, n_edges

class HitGraphDataset():
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir=None, filelist=None, n_samples=None, real_weight=1.0, false_weight=1.0,
                n_input_dir=1, input_dir2=None, input_dir3=None, random_permutation=True, n_samples2=None,
                n_samples3=None):

        self.random_permutation = random_permutation
        hits_info, edge_index, trig, e, n_hits, n_edges = load_file('/home/tingting/kdd2022/trigger_pred/data/mixed_data_7layer_260k.npz')

        self.hits_info = []
        self.edge_index = []
        # self.trig = trig
        self.e = []
        # self.n_hits = n_hits
        self.trig = []
        self.n_hits = []

        for index in range(n_samples):
            self.hits_info.append(hits_info[index][:n_hits[index]])
            self.edge_index.append(edge_index[index][:, :n_edges[index]])
            self.e.append(e[index][:n_edges[index]])
            self.trig.append(trig[index])
            self.n_hits.append(n_hits[index])

        for index in range(n_hits.shape[0]-1, n_hits.shape[0]-n_samples-1, -1):
            self.hits_info.append(hits_info[index][:n_hits[index]])
            self.edge_index.append(edge_index[index][:, :n_edges[index]])
            self.e.append(e[index][:n_edges[index]])

    def __getitem__(self, index):
        return torch_geometric.data.Data(x=torch.from_numpy(self.hits_info[index]), \
            edge_index=torch.from_numpy(self.edge_index[index].astype(np.long)),\
            trigger=torch.from_numpy(np.array(self.trig[index])), \
            n_hits=torch.from_numpy(np.array(self.n_hits[index])), \
            e=torch.from_numpy(np.array(self.e[index])))

    def __len__(self):
        return len(self.n_hits)

# def get_data_loaders(name, batch_size, model_name, **data_args):
#     if name == 'hits_loader':
#         train_dataset, valid_dataset, train_data_n_nodes, valid_data_n_nodes = get_datasets(**data_args)
#     else:
#         raise Exception('Dataset %s unknown' % name)
    
#     from torch_geometric.data import Batch
#     collate_fn = Batch.from_data_list
#     if model_name == 'GNNPairDiffpool':
#         train_batch_sampler = JetsBatchSampler(train_data_n_tracks, batch_size)
#         train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
#         valid_batch_sampler = JetsBatchSampler(valid_data_n_tracks, batch_size)
#         valid_data_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, collate_fn=collate_fn)
    
#     else:
#         train_batch_sampler = JetsBatchSampler(train_data_n_nodes, batch_size)
#         train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
#         valid_batch_sampler = JetsBatchSampler(valid_data_n_nodes, batch_size)
#         valid_data_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, collate_fn=collate_fn)
#     return train_data_loader, valid_data_loader


# def get_datasets(n_train, n_valid, input_dir=None, filelist=None, real_weight=1.0, false_weight=1.0,
#                 n_input_dir=1, input_dir2=None, input_dir3=None, random_permutation=True,
#                 n_train2=None, n_valid2=None, n_train3=None, n_valid3=None):
#     data = HitGraphDataset(input_dir=input_dir, filelist=filelist,
#                            n_samples=n_train+n_valid, real_weight=real_weight,
#                            false_weight=false_weight,
#                            n_input_dir=n_input_dir, input_dir2=input_dir2, 
#                            input_dir3=input_dir3, random_permutation=random_permutation,
#                            n_samples2=n_train2+n_valid2, n_samples3=n_train3+n_valid3)

#     # valid_rate = (n_valid+n_valid2) / (n_train+n_train2 + n_valid+n_valid2) 
#     # total_events = len(data) 
#     # valid_events = int(total_events * valid_rate)
#     # train_events = total_events - valid_events
#     # Split into train and validation
#     if n_input_dir == 2:
#         train_data, valid_data = random_split(data, [n_train+n_train2, n_valid+n_valid2])
#         # train_data, valid_data = random_split(data, [train_events, valid_events])
#     elif n_input_dir == 3:
#         train_data, valid_data = random_split(data, [n_train+n_train2+n_train3, n_valid+n_valid2+n_valid3])
#     else:
#         train_data, valid_data = random_split(data, [n_train, n_valid])

#     train_data_n_nodes = data.n_hits[train_data.indices]
#     valid_data_n_nodes = data.n_hits[valid_data.indices]
#     return train_data, valid_data, train_data_n_nodes, valid_data_n_nodes

# class JetsBatchSampler(Sampler):
#     def __init__(self, n_nodes_array, batch_size):
#         """
#         Initialization
#         :param n_nodes_array: array of sizes of the jets
#         :param batch_size: batch size
#         """
#         super().__init__(n_nodes_array.size)

#         self.dataset_size = n_nodes_array.size
#         self.batch_size = batch_size

#         self.index_to_batch = {}
#         self.node_size_idx = {}
#         running_idx = -1

#         for n_nodes_i in set(n_nodes_array):

#             if n_nodes_i <= 1:
#                 continue
#             self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

#             n_of_size = len(self.node_size_idx[n_nodes_i])
#             n_batches = np.ceil(max(n_of_size / self.batch_size, 1))

#             self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
#                                                            n_batches)
#             for batch in self.node_size_idx[n_nodes_i]:
#                 running_idx += 1
#                 self.index_to_batch[running_idx] = batch

#         self.n_batches = running_idx + 1

#     def __len__(self):
#         return self.n_batches

#     def __iter__(self):
#         batch_order = np.random.permutation(np.arange(self.n_batches))
#         for i in batch_order:
#             yield self.index_to_batch[i]



from collections import namedtuple
# System imports
import os
import random

# External imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, Sampler
import torch_geometric
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
def load_file(filename):

    with np.load(filename) as f:
        trigger_flag = f['trig']
        edge_index = f['edge_index']
        hits = f['hits_info']
        # hits = np.concatenate([hits, np.reshape([hits.shape[0]] * hits.shape[0], (-1, 1))], 1)
        e = f['e']
        n_hits = f['n_hits']
        n_edges = f['n_edges']

    return hits, edge_index, trigger_flag, e, n_hits, n_edges



def get_data_loaders(name, batch_size, model_name, **data_args):
    if name == 'hits_loader':
        train_dataset, valid_dataset, train_data_n_nodes, valid_data_n_nodes = get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)
    
    from torch_geometric.data import Batch
    collate_fn = Batch.from_data_list
    # if model_name == 'GNNPairDiffpool':
    #     train_batch_sampler = JetsBatchSampler(train_data_n_tracks, batch_size)
    #     train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    #     valid_batch_sampler = JetsBatchSampler(valid_data_n_tracks, batch_size)
    #     valid_data_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, collate_fn=collate_fn)
    
    # else:
    #     train_batch_sampler = JetsBatchSampler(train_data_n_nodes, batch_size)
    #     train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    #     valid_batch_sampler = JetsBatchSampler(valid_data_n_nodes, batch_size)
    #     valid_data_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, collate_fn=collate_fn)
    loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=16)
    train_data_loader = DataLoader(train_dataset, sampler=None,
                                   shuffle=True, **loader_args)
    valid_data_loader = (DataLoader(valid_dataset, sampler=None, **loader_args)
                         if valid_dataset is not None else None)
    return train_data_loader, valid_data_loader


def get_datasets(n_train, n_valid, input_dir=None, filelist=None, real_weight=1.0, false_weight=1.0,
                n_input_dir=1, input_dir2=None, input_dir3=None, random_permutation=True,
                n_train2=None, n_valid2=None, n_train3=None, n_valid3=None):
    data = HitGraphDataset(input_dir=input_dir, filelist=filelist,
                           n_samples=n_train+n_valid, real_weight=real_weight,
                           false_weight=false_weight,
                           n_input_dir=n_input_dir, input_dir2=input_dir2, 
                           input_dir3=input_dir3, random_permutation=random_permutation,
                           n_samples2=n_train2+n_valid2, n_samples3=n_train3+n_valid3)

    # valid_rate = (n_valid+n_valid2) / (n_train+n_train2 + n_valid+n_valid2) 
    # total_events = len(data) 
    # valid_events = int(total_events * valid_rate)
    # train_events = total_events - valid_events
    # Split into train and validation
    if n_input_dir == 2:
        train_data, valid_data = random_split(data, [n_train+n_train2, n_valid+n_valid2])
        # train_data, valid_data = random_split(data, [train_events, valid_events])
    elif n_input_dir == 3:
        train_data, valid_data = random_split(data, [n_train+n_train2+n_train3, n_valid+n_valid2+n_valid3])
    else:
        train_data, valid_data = random_split(data, [n_train, n_valid])

    train_data_n_nodes = data.n_hits[train_data.indices]
    valid_data_n_nodes = data.n_hits[valid_data.indices]
    return train_data, valid_data, train_data_n_nodes, valid_data_n_nodes

# class JetsBatchSampler(Sampler):
#     def __init__(self, n_nodes_array, batch_size):
#         """
#         Initialization
#         :param n_nodes_array: array of sizes of the jets
#         :param batch_size: batch size
#         """
#         super().__init__(n_nodes_array.size)

#         self.dataset_size = n_nodes_array.size
#         self.batch_size = batch_size

#         self.index_to_batch = {}
#         self.node_size_idx = {}
#         running_idx = -1

#         for n_nodes_i in set(n_nodes_array):

#             if n_nodes_i <= 1:
#                 continue
#             self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

#             n_of_size = len(self.node_size_idx[n_nodes_i])
#             n_batches = np.ceil(max(n_of_size / self.batch_size, 1))

#             self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
#                                                            n_batches)
#             for batch in self.node_size_idx[n_nodes_i]:
#                 running_idx += 1
#                 self.index_to_batch[running_idx] = batch

#         self.n_batches = running_idx + 1

#     def __len__(self):
#         return self.n_batches

#     def __iter__(self):
#         batch_order = np.random.permutation(np.arange(self.n_batches))
#         for i in batch_order:
#             yield self.index_to_batch[i]
