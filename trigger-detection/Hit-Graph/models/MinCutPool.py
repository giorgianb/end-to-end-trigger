from builtins import print
from main_script.main_for_tracking_result import DEVICE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.vp_GNN import GNN
from models.encoders import SoftPoolingGcnEncoder
from .utils import make_mlp, preprocess_adj
from torch_geometric.nn.dense.mincut_pool import dense_mincut_pool
from torch_geometric.nn import GCNConv, DenseGCNConv, global_mean_pool

class MinCutDiffpool(nn.Module):
    def __init__(self, mincut_dim, hidden_dim, hidden_activation='Tanh', layer_norm=True, GNN_config={}, learning_rate=0.001, lr_scheduler_decrease_rate=0.95):
        """
        SetToGraph model.
        :param in_features: input set's number of features per data point
        :param out_features: number of output features.
        :param set_fn_feats: list of number of features for the output of each deepsets layer
        :param method: transformer method - quad, lin2 or lin5
        :param hidden_mlp: list[int], number of features in hidden layers mlp.
        :param predict_diagonal: Bool. True to predict the diagonal (diagonal needs a separate psi function).
        :param attention: Bool. Use attention in DeepSets
        :param cfg: configurations of using second bias in DeepSetLayer, normalization method and aggregation for lin5.
        """
        super(MinCutDiffpool, self).__init__()

        self.name = 'MinCutPool'

        #input model
        self.input_network = make_mlp(4, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)

        self.loss_func = nn.BCELoss()

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.assign1 = make_mlp(hidden_dim, [hidden_dim, mincut_dim[0]], hidden_activation=hidden_activation, layer_norm=layer_norm)

        self.conv4 = DenseGCNConv(hidden_dim, hidden_dim)
        self.conv5 = DenseGCNConv(hidden_dim, hidden_dim)
        self.conv6 = DenseGCNConv(hidden_dim, hidden_dim)

        self.assign2 = make_mlp(hidden_dim, [hidden_dim, mincut_dim[1]], hidden_activation=hidden_activation, layer_norm=layer_norm)

        # diffpool model
        # self.diffpool = SoftPoolingGcnEncoder(input_dim=hidden_dim, **diff_pool_config)
        self.output_network = make_mlp(hidden_dim, [hidden_dim, hidden_dim, 1],
                                      hidden_activation=hidden_activation,
                                      output_activation=None,
                                      layer_norm=layer_norm)

        # optimizer init
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        # lr_scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)
        
                
    def forward(self, x, edge_index, batch, batch_size, e=None, is_e=False):
        x = self.input_network(x)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)

        s = self.assign1(x)

        # print(node_output)
        n_hits = x.shape[0] // batch_size
        x = x.view(batch_size.item(), n_hits, x.shape[1])
        s = s.view(batch_size.item(), n_hits, s.shape[1])
        A = torch.cuda.FloatTensor(batch_size, n_hits, n_hits).fill_(0)
        start, end = edge_index
        if is_e:
            A[batch[start], (start-n_hits*batch[start]), (end-n_hits*batch[start])] = e.reshape(-1)
        else:
            A[batch[start], (start-n_hits*batch[start]), (end-n_hits*batch[start])] = 1

        A = preprocess_adj(A)

        out, out_adj, mincut_loss1, ortho_loss1 = dense_mincut_pool(x, A, s)

        x = self.conv4(out, out_adj).relu()
        x = self.conv5(x, out_adj).relu()
        x = self.conv6(x, out_adj)

        s = self.assign2(x)

        out, out_adj, mincut_loss2, ortho_loss2 = dense_mincut_pool(x, out_adj, s)

        pool_batch = torch.tensor(np.repeat(np.arange(out.shape[0]), out.shape[1]), dtype=torch.long, device=DEVICE)
        out = out.view(-1, out.shape[2])
        graph_pred = self.output_network(global_mean_pool(out, pool_batch))
        return graph_pred, mincut_loss1, ortho_loss1, mincut_loss2, ortho_loss2

    def train_model(self, ip_pred, ip_true, mincut_loss1, ortho_loss1, mincut_loss2, ortho_loss2):
        self.optimizer.zero_grad()
        loss = self.get_loss(ip_pred, ip_true, mincut_loss1, ortho_loss1, mincut_loss2, ortho_loss2)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_loss(self, ip_pred, ip_true, mincut_loss1, ortho_loss1, mincut_loss2, ortho_loss2):
        sigmoid = nn.Sigmoid()
        # print(self.loss_func(sigmoid(ip_pred), ip_true), mincut_loss, ortho_loss)
        return self.loss_func(sigmoid(ip_pred), ip_true) + mincut_loss1 + ortho_loss1 + mincut_loss2 + ortho_loss2