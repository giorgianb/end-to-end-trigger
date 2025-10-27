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
from .layers import Masked_LayerNorm
from .Bipartite_Attention_gLRI import get_approximate_radii, get_track_endpoints, get_predicted_pz

class FastML(nn.Module):
    def __init__(
            self,
            num_features,
            num_classes,
            layers_spec, # List of tuple of ((feature_dim, n_aggregators))
            hidden_activation='Tanh', 
            aggregator_activation='softmax',
            recalculate_hits_mean=True,
            ):
        super().__init__()
        self.recalculate_hits_mean = recalculate_hits_mean

        self.t1 = TransformerBlock(num_features)
        self.t2 = TransformerBlock(6)
        self.t3 = TransformerBlock(6)
        self.f1 = nn.Linear(6, 32)
        self.f2 = nn.Linear(32, 16)
        self.f3 = nn.Linear(16, num_classes)

        self._track_project = nn.Sequential(
                nn.Linear(6, 2*6),
                getattr(nn, hidden_activation)(),
                nn.Linear(2*6, 6)
        )

    def forward(self, X, mask):
        X, pred = self.generate_track_embeddings(X, mask)
        return X


    def generate_track_embeddings(self, X, mask):
        if self.recalculate_hits_mean:
            X = self._recalculate_hits_mean(X, mask)

        X = self.t1(X, mask)
        X = self.t2(X, mask)
        X = self.t3(X, mask)

        pred = self._masked_mean(X, mask)
        pred = self.f1(pred)
        pred = self.f2(pred)
        pred = self.f3(pred)

        return X, pred

    @staticmethod
    def _recalculate_hits_mean(X, mask):
        # this will ensure all masked tracks have 0 in their hits
        # shape (minibatch, track, 15)
        Xp = torch.zeros_like(X)
        Xp[:, :, :] = X[:, :, :]
        hits = X[:, :, :15] * mask.unsqueeze(-1)

        # (minibatch, track, layer, coords)
        hits = hits.reshape((X.shape[0], X.shape[1], 5, 3))

        # (minibatch, coords)
        total = torch.sum(hits, dim=(1, 2))

        # (minibatch, track, layer)
        good_hits = torch.all(hits != 0, dim=-1)

        # (minibatch,)
        n_good_hits = torch.sum(good_hits * mask.unsqueeze(-1), dim=(1, 2))
        n_good_hits[n_good_hits == 0] = 1

        hits_mean = total / n_good_hits.unsqueeze(-1)
        Xp[:, :, (15+10):(15+13)] = hits_mean.unsqueeze(1)
        Xp= Xp* mask.unsqueeze(-1)

        return Xp



    def predict_adjacency_matrix(self, H_t, mask):
        H_t = self._track_project(H_t)
        # H_t: (batch, track, n)
        A = torch.sum(H_t.unsqueeze(-2) * H_t.unsqueeze(-3), dim=-1) 
        A = A * (mask.unsqueeze(-1) * mask.unsqueeze(-2))
        return A

    @staticmethod
    def _masked_mean(x, mask):
        # : (minibatch, track, feature)
        xp = x * mask.unsqueeze(-1)
        summed = torch.sum(xp, dim=1) # (batch, feature)
        # mask: (batch, track
        n_tracks = torch.sum(mask, dim=-1).reshape(summed.shape[0], 1)
        n_tracks[n_tracks == 0] = 1
        return summed / n_tracks





class TransformerBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mab = Masked_SAB(input_dim, 32, 2, ln=False)
        self.dense = nn.Sequential(
            nn.Linear(32, 8),
            nn.Linear(8, 6)
        )
        # Project input dim to 6 using 1x1 conv
        self.project = nn.Conv1d(input_dim, 6, 1)

    def forward(self, X, mask):
        Xi = self.mab(X, mask)
        Xi = self.dense(Xi)
        Xi = Xi + self.project(X.transpose(-1, -2)).transpose(-1, -2)
        return Xi * mask.unsqueeze(-1)


