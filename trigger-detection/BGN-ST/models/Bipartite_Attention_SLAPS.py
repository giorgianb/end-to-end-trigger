from os import X_OK
from sklearn import utils
from torch import tensor
from torch._C import is_anomaly_enabled
import torch.nn as nn
import torch
from typing import OrderedDict, Tuple
from icecream import ic
from torch.nn.modules import distance
from torch_scatter import scatter_add

from random import sample
import logging
from icecream import ic
import sys

# local
from .utils_settrans import Masked_SAB

import torch_geometric.nn as gnn

from .Bipartite_Attention_Masked import Bipartite_Attention as Bipartite_Attention_Masked
class Bipartite_Attention(nn.Module):
    def __init__(
            self,
            num_features,
            num_classes,
            layers_spec, # List of tuple of ((feature_dim, n_aggregators))
            adj_embeddings_size=128,
            hidden_activation='Tanh', 
            aggregator_activation='softmax',
            ln=False,
            bn=False,
            recalculate_hits_mean=True,
            self_split=False,
            corruption_probability = 0.2,
            n_neighbors = 5, # So we can easily get fourthness (room for thirdness and then some)
            sag_pool=True
            ):
        super().__init__()

        self.embeddings_generator = Bipartite_Attention_Masked(
                num_features,
                num_classes,
                layers_spec,
                hidden_activation=hidden_activation,
                aggregator_activation=aggregator_activation,
                ln=ln,
                bn=bn,
                recalculate_hits_mean=recalculate_hits_mean,
                self_split=self_split,
                final_pooling=True
        )

        self.mlp_knn = MLP_KNN(
                layers_spec[-1][0], 
                adj_embeddings_size, 
                n_neighbors, 
                hidden_activation=hidden_activation, 
                bn=bn)
        self.corruption_probability = corruption_probability
        activation = getattr(nn, hidden_activation)

        self.denoiser = gnn.Sequential('x, edge_index', [
                (gnn.GATv2Conv(layers_spec[-1][0], layers_spec[-1][0], heads=4), 'x, edge_index -> x'),
                nn.Linear(4*layers_spec[-1][0], layers_spec[-1][0]),
                activation(),
                (gnn.GATv2Conv(layers_spec[-1][0], layers_spec[-1][0], heads=4), 'x, edge_index -> x'),
                nn.Linear(4*layers_spec[-1][0], layers_spec[-1][0]),
                activation(),
            ])

        final_pooling =  (gnn.SAGPooling(layers_spec[-1][0], 1), 'x, edge_index, batch -> x') if sag_pool else (gnn.global_mean_pool, 'x, batch -> x')


        self.classifier = gnn.Sequential('x, edge_index, batch', [
                (gnn.GATv2Conv(layers_spec[-1][0], layers_spec[-1][0], heads=4), 'x, edge_index -> x'),
                nn.Linear(4*layers_spec[-1][0], layers_spec[-1][0]),
                activation(),
                (gnn.GATv2Conv(layers_spec[-1][0], layers_spec[-1][0], heads=4), 'x, edge_index -> x'),
                nn.Linear(4*layers_spec[-1][0], layers_spec[-1][0]),
                activation(),
                final_pooling,
                nn.Linear(layers_spec[-1][0], num_classes)
        ])

        self.num_classes = num_classes



    def forward(self, X, mask):
        node_embeddings = self.embeddings_generator._generate_track_embeddings(X, mask)
        edges = self.mlp_knn(node_embeddings, mask)
        node_embeddings = node_embeddings[mask.to(torch.bool)]
        batch = torch.arange(X.shape[0], device=X.device).unsqueeze(-1).repeat(1, X.shape[1])[mask.to(torch.bool)]
        batch = torch.cumsum(batch[:-1] != batch[1:], dim=0)
        batch = torch.cat([torch.tensor([0]).to(X.device), batch], dim=0)
        partial_labels = self.classifier(node_embeddings, edges, batch)
        labels = torch.zeros((X.shape[0], self.num_classes), device=X.device)
        non_empty = torch.sum(mask, dim=-1) != 0
        labels[non_empty] = partial_labels

        corruption_mask = torch.rand(X.shape) <= self.corruption_probability
        Xp = torch.clone(X)
        Xp[corruption_mask] = 0
        corrupted_embeddings = self.embeddings_generator._generate_track_embeddings(Xp, mask)
        corrupted_embeddings = corrupted_embeddings[mask.to(torch.bool)]
        reconstructed = self.denoiser(corrupted_embeddings, edges)

        return labels, node_embeddings, reconstructed

    def generate_embedding(self, X, mask):
        return self.embeddings_generator.generate_embedding(X, mask)

class MLP_KNN(nn.Module):
    def __init__(
            self,
            num_features,
            embeddings_size, 
            n_neighbors,
            hidden_activation='Tanh',
            bn=False
        ):
        super().__init__()
        self.n_neighbors = n_neighbors

        activation = getattr(nn, hidden_activation)
        self.embeddings_generator = nn.ModuleList([
                nn.Linear(num_features, 2*embeddings_size),
                activation(),
                nn.Linear(2*embeddings_size, embeddings_size),
                activation()
        ])

    def forward(self, X, mask):
        for layer in self.embeddings_generator:
            X = layer(X) * mask.unsqueeze(-1)

            
        mask = mask.to(torch.bool)
        delta = X.unsqueeze(-2) - X.unsqueeze(1)
        distances = torch.sum(delta**2, dim=-1)
        distances_mask = mask.unsqueeze(1).repeat(1, X.shape[1], 1)
        distances_mask = distances_mask.to(torch.bool)
        # Set columns of empty tracks to infinity
        distances[~distances_mask] = torch.inf
        # Set rows of empty tracks to infinity
        distances[~mask.to(torch.bool)] = torch.inf

        d, edges_per_batch = torch.sort(distances, dim=-1)
        edges_per_batch = edges_per_batch[:, :, :self.n_neighbors]
        d = d[:, :, :self.n_neighbors]
        n_tracks = torch.sum(mask, dim=-1)
        track_offset = torch.cumsum(n_tracks, dim=0)
        track_offset = torch.roll(track_offset, 1)
        track_offset[0] = 0

        node_names = torch.cumsum(distances != torch.inf, dim=-1)
        node_names = torch.roll(node_names, 1, dims=-1)
        node_names[:, :, 0] = 0
        node_names += track_offset.reshape(-1, 1, 1)


        edges_per_batch = torch.gather(node_names, -1, edges_per_batch)

        track_index = torch.cumsum(mask, dim=-1)
        track_index = torch.roll(track_index, 1, dims=-1)
        track_index[:, 0] = 0
        track_index += track_offset.unsqueeze(-1)
        track_index = track_index.unsqueeze(-1).repeat(1, 1, self.n_neighbors)

        edges = torch.stack([track_index[d != torch.inf], edges_per_batch[d != torch.inf]], dim=0)

        return edges

