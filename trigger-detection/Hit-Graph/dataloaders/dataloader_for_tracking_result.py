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
def load_graph(filename):

    with np.load(filename) as f:
        trigger_flag = f['trigger']
        edge_index = f['edge_index']
        hits = f['hits']
        # hits = np.concatenate([hits, np.reshape([hits.shape[0]] * hits.shape[0], (-1, 1))], 1)
        e = f['e']

    return hits, edge_index, trigger_flag, e

class HitGraphDataset():
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, filenames):
        self.filenames = filenames
        random.shuffle(self.filenames)
        self.hits_info = []
        self.edge_index = []
        self.trig = []
        self.e = []
        self.n_hits = []

        for file_index in range(len(self.filenames)):
            hits, edge_index, trigger_flag, e = load_graph(self.filenames[file_index])
            self.hits_info.append(hits)
            self.edge_index.append(edge_index)
            self.trig.append(trigger_flag)
            self.e.append(e)
            self.n_hits.append(hits.shape[0])
        self.n_hits = np.array(self.n_hits)
    

    def __getitem__(self, index):
        return torch_geometric.data.Data(x=torch.from_numpy(self.hits_info[index]), \
            edge_index=torch.from_numpy(self.edge_index[index].astype(np.long)),\
            trigger=torch.from_numpy(self.trig[index]), \
            n_hits=torch.from_numpy(np.array(self.n_hits[index])), \
            e=torch.from_numpy(np.array(self.e[index])))

    def __len__(self):
        return len(self.n_hits)

def get_data_loaders(name, batch_size, model_name, **data_args):
    if name == 'new_hits_loader':
        train_dataset, valid_dataset, test_dataset = get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)
    
    from torch_geometric.data import Batch
    collate_fn = Batch.from_data_list
    if model_name == 'GNNPairDiffpool':
        train_batch_sampler = JetsBatchSampler(train_dataset.n_tracks, batch_size)
        train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
        valid_batch_sampler = JetsBatchSampler(valid_dataset.n_tracks, batch_size)
        valid_data_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, collate_fn=collate_fn)
        test_batch_sampler = JetsBatchSampler(test_dataset.n_tracks, batch_size)
        test_data_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, collate_fn=collate_fn)
    
    else:
        train_batch_sampler = JetsBatchSampler(train_dataset.n_hits, batch_size)
        train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
        valid_batch_sampler = JetsBatchSampler(valid_dataset.n_hits, batch_size)
        valid_data_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, collate_fn=collate_fn)
        test_batch_sampler = JetsBatchSampler(test_dataset.n_hits, batch_size)
        test_data_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, collate_fn=collate_fn)
    
    return train_data_loader, valid_data_loader, test_data_loader


def get_datasets(n_train, n_valid, n_test, input_dir1, input_dir2):
    
    filenames1 = sorted([os.path.join(input_dir1, f) for f in os.listdir(input_dir1)
                                if f.startswith('event') and not f.endswith('_ID.npz')])[:n_train+n_valid+n_test]
    filenames2 = sorted([os.path.join(input_dir2, f) for f in os.listdir(input_dir2)
                                if f.startswith('event') and not f.endswith('_ID.npz')])[:n_train+n_valid+n_test]
    train_filenames = filenames1[:n_train] + filenames2[:n_train]
    valid_filenames = filenames1[n_train: n_train + n_valid] + filenames2[n_train: n_train + n_valid]
    test_filenames = filenames1[n_train + n_valid : ] + filenames2[n_train + n_valid:]

    train_data = HitGraphDataset(filenames = train_filenames)
    valid_data = HitGraphDataset(filenames = valid_filenames)
    test_data = HitGraphDataset(filenames = test_filenames)

    return train_data, valid_data, test_data

class JetsBatchSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size):
        """
        Initialization
        :param n_nodes_array: array of sizes of the jets
        :param batch_size: batch size
        """
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            if n_nodes_i <= 1:
                continue
            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

            n_of_size = len(self.node_size_idx[n_nodes_i])
            n_batches = np.ceil(max(n_of_size / self.batch_size, 1))

            self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
                                                           n_batches)
            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]
