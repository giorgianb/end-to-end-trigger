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
# from utils import isParallelMatrix, shortest_dist_parallel_matrix, shortest_dist_non_parallel_matrix

from random import sample
import logging
from icecream import ic
import sys

# local
from .utils_settrans import Masked_SAB

class Bipartite_Attention(nn.Module):
    def __init__(
            self,
            num_features,
            num_classes,
            layers_spec, # Tuple of (N, feature_dim, coordinate_dim)
            hidden_activation='Tanh', 
            aggregator_activation='potential',
            ln=False,
            dropout=False,
            dropout_rate=0,
            ):
        super(Bipartite_Attention, self).__init__()
        _layers = []
        prev_dim = num_features
        
        for feature_dim, n_aggregators in layers_spec:
            _layers.append(
                    Bipartite_Layers(
                        prev_dim,
                        feature_dim,
                        n_aggregators,
                        hidden_activation,
                        aggregator_activation,
                    )
            )
            prev_dim = feature_dim 

        self._layers = nn.ModuleList(_layers)

        # self._layers = nn.Sequential(
        #         *garnet_layers
        # )

        self._pred_layers = nn.Sequential(
                nn.Linear(2*prev_dim, 2*prev_dim),
                nn.ReLU(),
                nn.Linear(2*prev_dim, num_classes)
        )

        self.is_dropout = dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X, mask):
        def masked_mean(x):
            # : (minibatch, track, feature)
            xp = x * mask.unsqueeze(-1)
            summed = torch.sum(x, dim=-2) # (batch, features)
            # mask: (batch, track)
            n_tracks = torch.sum(mask, dim=-1).reshape(summed.shape[0], 1)
            n_tracks[n_tracks == 0] = 1
            return summed / n_tracks
        
        if self.dropout:
            X[:, :, :15] = self.dropout(X[:, :, :15])
            
        for layer in self._layers:
            pred_x, agg = layer(X, mask)
            X = pred_x

        mean_pooled = masked_mean(pred_x)
        min_elem = torch.min(pred_x, dim=-2)[0] - 1
        x_premaxpool = pred_x*mask.unsqueeze(-1) + (1 - mask).unsqueeze(-1)*(min_elem.unsqueeze(-2)) # set all masked edges -torch.inf
        #max_pooled = torch.max(x_premaxpool, dim=-2)[1] # TODO: change back. We are only using the indices experimentally. It seems to provide better performance but it breaks permutation invariance and leads to different results between hit-tracks and pixel-tracks
        max_pooled = torch.max(x_premaxpool, dim=-2)[0] 

        H = torch.cat((mean_pooled, max_pooled), dim=-1)
        return self._pred_layers(H)

class Bipartite_Layers(nn.Module):
    def __init__(self,
        input_dim,
        feature_dim,
        n_aggregators,
        hidden_activation,
        aggregator_activation,
        num_heads=4,
    ):
        super(Bipartite_Layers, self).__init__()
        self.aggregator_activation = aggregator_activation
        self.enc = nn.ModuleList([
            Masked_SAB(input_dim, feature_dim, num_heads),
            Masked_SAB(feature_dim, feature_dim, num_heads),
        ])
        self.transform_in = nn.Linear(feature_dim, feature_dim)
        self.aggregator_score = nn.Linear(feature_dim, n_aggregators)
        self.transform_out = nn.Sequential(
                nn.Linear(2*feature_dim*n_aggregators + input_dim + feature_dim, feature_dim),
                getattr(nn, hidden_activation)()
        )
        self._feature_dim = feature_dim
        self._n_aggregators = n_aggregators

    def forward(self, X, mask):
        """
        X: Tensor of shape [n_minibatches, n_tracks, n_track_features]
        """
        Xp = self.enc[0](X, mask) * mask.unsqueeze(-1)
        Xp = self.enc[1](Xp, mask) * mask.unsqueeze(-1)
        Xp = self.transform_in(Xp) * mask.unsqueeze(-1)
        attention_score = self.aggregator_score(Xp)

        def masked_softmax(x):
            max_elem = torch.max(x, dim=1)[0]
            xp = x - max_elem.unsqueeze(1)
            num = torch.exp(xp)*mask.unsqueeze(-1)
            dem = torch.sum(num, dim=1).unsqueeze(1)
            return num/(dem + 1e-16)

        def masked_mean(x):
            # : (minibatch, aggregator, feature, track)
            xp = x * mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
            summed = torch.sum(xp, dim=-1) # (batch, aggregator, feature)
            # mask: (batch, track
            n_tracks = torch.sum(mask, dim=-1).reshape(summed.shape[0], 1, 1)
            n_tracks[n_tracks == 0] = 1
            return summed / n_tracks
        
        if self.aggregator_activation == 'potential':
            attention_score = torch.exp(-torch.abs(attention_score)) * mask.unsqueeze(-1)
        elif self.aggregator_activation == 'ReLU':
            act = nn.ReLU()
            attention_score = act(attention_score) * mask.unsqueeze(-1)
        elif self.aggregator_activation == 'Tanh':
            act = nn.Tanh()
            attention_score = act(attention_score) * mask.unsqueeze(-1)
        elif self.aggregator_activation == 'softmax':
            attention_score = masked_softmax(attention_score)
        else:
            attention_score = attention_score * mask.unsqueeze(-1)
            # (batch, n_tracks, n_aggregators)
            # (batch, n_traccks)
        edges = torch.einsum('btf,bta->baft', Xp, attention_score)
        min_elem = torch.min(edges, dim=-1)[0] - 1
        edges_premaxpool = edges*mask.reshape(mask.shape[0], 1, 1, mask.shape[1]) + (1 - mask).reshape(mask.shape[0], 1, 1, mask.shape[1])*(min_elem.unsqueeze(-1)) # set all masked edges -torch.inf
        max_pooled = torch.max(edges_premaxpool, dim=-1)[0] # (n_minibatches, n_aggregators, n_features)
        mean_pooled = masked_mean(edges)
        aggregator = torch.cat((max_pooled, mean_pooled), axis=-1)
        aggregator = aggregator.reshape(aggregator.shape[0], aggregator.shape[1]*aggregator.shape[2])
        agg_repeated = aggregator.reshape(aggregator.shape[0], 1, -1).repeat(1, X.shape[1], 1)
        H = torch.cat((X, Xp, agg_repeated), axis=-1)

        return self.transform_out(H)*mask.unsqueeze(-1), aggregator

    def forward_check(self, X, Xp, mask):
        """
        X: Tensor of shape [n_minibatches, n_tracks, n_track_features]
        """

        def compare(X, X_old, track_axis=1):
            X = torch.sort(X, axis=1)[0]
            X_old = torch.sort(X_old, axis=track_axis)[0]
            return torch.all(torch.isclose(X, X_old))

        add = len(self.Xp_old) == 0
        index = 0

        Xp = self.transform_in(Xp) * mask.unsqueeze(-1)
        attention_score = self.aggregator_score(Xp)
        if add and self.check:
            self.Xp_old.append(attention_score)
        elif self.check:
            assert compare(attention_score, self.Xp_old[index]), f"X: {attention_score}\n X_old: {self.Xp_old[index]}"
            index += 1


        def masked_softmax(x):
            max_elem = torch.max(x)
            xp = x - max_elem
            num = torch.exp(xp)*mask.unsqueeze(-1)
            dem = torch.sum(num, dim=1).unsqueeze(1)
            return num/(dem + 1e-16)

        def masked_mean(x):
            # : (minibatch, aggregator, feature, track)
            xp = x * mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
            summed = torch.sum(xp, dim=-1) # (batch, aggregator, feature)
            # mask: (batch, track
            n_tracks = torch.sum(mask, dim=-1).reshape(summed.shape[0], 1, 1)
            n_tracks[n_tracks == 0] = 1
            return summed / n_tracks


        attention_score = masked_softmax(attention_score)
        if add and self.check:
            self.Xp_old.append(attention_score)
        elif self.check:
            assert compare(attention_score, self.Xp_old[index]), f"X: {attention_score}\n X_old: {self.Xp_old[index]}"
            index += 1


        edges = torch.einsum('btf,bta->baft', Xp, attention_score)
        if add and self.check:
            self.Xp_old.append(edges)
        elif self.check:
            assert compare(edges, self.Xp_old[index]), f"X: {edges}\n X_old: {self.Xp_old[index]}"
            index += 1



        min_elem = torch.min(edges) - 1
        edges_premaxpool = edges*mask.reshape(mask.shape[0], 1, 1, mask.shape[1]) + (1 - mask).reshape(mask.shape[0], 1, 1, mask.shape[1])*(min_elem) # set all masked edges -torch.inf
        max_pooled = torch.max(edges_premaxpool, dim=-1)[0] # (n_minibatches, n_aggregators, n_features)
        if add and self.check:
            self.Xp_old.append(max_pooled)
        elif self.check:
            assert compare(max_pooled, self.Xp_old[index]), f"X: {max_pooled}\n X_old: {self.Xp_old[index]}"
            index += 1


        mean_pooled = masked_mean(edges)
        if add and self.check:
            self.Xp_old.append(mean_pooled)
        elif self.check:
            assert compare(mean_pooled, self.Xp_old[index]), f"X: {mean_pooled}\n X_old: {self.Xp_old[index]}"
            index += 1

        aggregator = torch.cat((max_pooled, mean_pooled), axis=-1)
        if add and self.check:
            self.Xp_old.append(aggregator)
        elif self.check:
            assert compare(aggregator, self.Xp_old[index]), f"X: {aggregator}\n X_old: {self.Xp_old[index]}"
            index += 1

        aggregator = aggregator.reshape(aggregator.shape[0], aggregator.shape[1]*aggregator.shape[2])
        agg_repeated = aggregator.reshape(aggregator.shape[0], 1, -1).repeat(1, Xp.shape[1], 1)
        H = torch.cat((X, Xp, agg_repeated), axis=-1)
        if add and self.check:
            self.Xp_old.append(H)
        elif self.check:
            assert compare(H, self.Xp_old[index]), f"X: {H}\n X_old: {self.Xp_old[index]}"
            index += 1



        return self.transform_out(H)*mask.unsqueeze(-1), aggregator
