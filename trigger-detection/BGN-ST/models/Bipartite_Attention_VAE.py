from sklearn import utils
from torch import tensor
from torch._C import is_anomaly_enabled
import torch.nn as nn
import torch
from typing import OrderedDict, Tuple
from icecream import ic
from torch.nn.modules import distance
from torch_scatter import scatter_add
import torch.nn.functional as F

from random import sample
import logging
from icecream import ic
import sys

# local
from .utils_settrans import Masked_SAB
from .layers import Masked_LayerNorm
from .Bipartite_Attention_Masked import Bipartite_Attention

class Bipartite_Attention_VAE(nn.Module):
    def __init__(
            self,
            num_features,
            num_gt_features,
            latent_dim,
            layers_spec, # List of tuple of ((feature_dim, n_aggregators))
            hidden_activation='Tanh', 
            aggregator_activation='softmax',
            ln=False,
            bn=False,
            recalculate_hits_mean=True,
            final_pooling=True,
            cylindrical_coordinates=False,
            ):
        super().__init__()
        self.cylindrical_coodinates = cylindrical_coordinates
        self.latent_dim = latent_dim
        self.encoder = Bipartite_Attention(
                # +1 for the label
                num_features=(num_features + num_gt_features + 1),
                num_classes=(latent_dim + latent_dim + 1),
                layers_spec=layers_spec,
                hidden_activation=hidden_activation,
                aggregator_activation=aggregator_activation,
                ln=ln,
                bn=bn,
                final_pooling=False,
                recalculate_hits_mean=recalculate_hits_mean
        )

        self.decoder = Bipartite_Attention(
                # +1 for the label
                num_features=(latent_dim + 1),
                num_classes=10 if self.cylindrical_coodinates else 15,
                layers_spec=layers_spec,
                hidden_activation=hidden_activation,
                aggregator_activation=aggregator_activation,
                ln=ln,
                bn=bn,
                final_pooling=False,
                recalculate_hits_mean=False,
        )
        if self.cylindrical_coodinates:
            self.r = nn.Parameter(torch.Tensor([2.53654013, 3.33592798, 4.1445396 , 7.46767887, 9.97344009]), requires_grad=False)

        hidden_activation = getattr(nn, hidden_activation)
        if ln:
            self.prior_params = nn.Sequential(
                    nn.Linear(num_features + 1, layers_spec[0][0]),
                    nn.LayerNorm(layers_spec[0][0]),
                    hidden_activation(),
                    nn.Linear(layers_spec[0][0], layers_spec[0][0]),
                    nn.LayerNorm(layers_spec[0][0]),
                    hidden_activation(),
                    nn.Linear(layers_spec[0][0], 2*latent_dim),
                )
        else:
            self.prior_params = nn.Sequential(
                    nn.Linear(num_features + 1, layers_spec[0][0]),
                    hidden_activation(),
                    nn.Linear(layers_spec[0][0], layers_spec[0][0]),
                    hidden_activation(),
                    nn.Linear(layers_spec[0][0], 2*latent_dim),
                )



    def forward(self, x, gt_x, labels, mask):
        counterfactual = (1 - labels.reshape(-1, 1, 1).repeat(1, x.shape[1], 1))
        dist_prior = self.prior_params(torch.cat([x, counterfactual], dim=-1))
        mu_prior, sigma_prior = dist_prior[:, :, :self.latent_dim], dist_prior[:, :, self.latent_dim:]
        x_f = torch.cat([x, gt_x, counterfactual], dim=-1)
        dist = self.encoder(x_f, mask)
        mu = dist[:, :, :self.latent_dim]
        # Do we scale by sigma or sigma**2
        sigma = dist[:, :, self.latent_dim:2*self.latent_dim]
        keep_prob = dist[:, :, 2*self.latent_dim:]
        keep_prob_logits = torch.cat([torch.zeros_like(keep_prob), keep_prob], dim=-1)
        new_mask = F.gumbel_softmax(keep_prob_logits, tau=1, hard=(1-self.training))[..., 1] * mask
        z = mu + torch.exp(0.5*sigma) * torch.randn_like(mu)
        z_full = torch.cat([z, counterfactual], dim=-1)
        x_rc = self.decoder(z_full, mask)
        if self.cylindrical_coodinates:
            hits = x_rc.reshape(-1, x_rc.shape[1], 5, 2)
            phi = hits[..., 0]
            z = hits[..., 1]
            #x = self.r.reshape(1, 1, -1) * torch.cos(phi)
            #y = self.r.reshape(1, 1, -1) * torch.sin(phi)
            x = self.r.reshape(1, 1, -1) * torch.cos(phi)
            y = self.r.reshape(1, 1, -1) * torch.sin(phi)
            x_rc = torch.stack([x, y, z], dim=-1).reshape(-1, x_rc.shape[1], 15)  * mask.unsqueeze(-1)
            print(f'{torch.any(torch.isnan(x_rc))=}')

        return x_rc, mu, sigma, new_mask, mu_prior, sigma_prior

    def get_representation(self, x, gt_x, labels, mask):
        counterfactual = labels.reshape(-1, 1, 1).repeat(1, x.shape[1], 1)
        x_f = torch.cat([x, gt_x, counterfactual], dim=-1)
        dist = self.encoder(x_f, mask)
        mu = dist[:, :, :self.latent_dim]
        sigma = dist[:, :, self.latent_dim:2*self.latent_dim]
        return mu, sigma

