import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from .utils import make_mlp

class GraphConv(nn.Module):
    '''y = normalize(W(Ax + x) + bias), or simplied version: y = W(Ax)'''
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        # print(f'y1: {y}')
        if self.add_self:
            y += x
        # print(f'y2: {y}')
        y = torch.matmul(y,self.weight)
        # print(f'y3: {y}')
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            #print(y[0][0])
        return y

class GraphConvPooling(nn.Module):
    def __init__(self, learning_rate=0.05, agg='sum', hidden_activation='Tanh',layer_norm=False, lr_scheduler_decrease_rate=0.995, **karg) -> None:
        super().__init__()
        self.name = 'GCNPooling'
        self.gcn = GraphConv(**karg)
        hidden_dim = karg['output_dim']
        self.outputnet = make_mlp(hidden_dim, [hidden_dim, hidden_dim, 1],
                                      hidden_activation=hidden_activation,
                                      output_activation=None,
                                      layer_norm=layer_norm)
        self.agg = getattr(torch, agg)
        # optimizer init
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        # lr_scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)
        loss_func='binary_cross_entropy_with_logits'
        self.loss_func = getattr(nn.functional, loss_func)

    def forward(self, x, edge_index, batch, batch_size, e=None, is_e=False):
        n_hits = x.shape[0] // batch_size
        nodes = x.view(batch_size.item(), n_hits, x.shape[1])
        A = torch.cuda.FloatTensor(batch_size, n_hits, n_hits).fill_(0)
        start, end = edge_index
        if is_e:
            A[batch[start], (start-n_hits*batch[start]), (end-n_hits*batch[start])] = e.reshape(-1)
        else:
            A[batch[start], (start-n_hits*batch[start]), (end-n_hits*batch[start])] = 1
        # print(nodes)
        nodes = self.gcn(nodes, A)
        # print(nodes)
        return self.outputnet(self.agg(nodes, dim=1)[0])

    def train_model(self, ip_pred, ip_true):
        self.optimizer.zero_grad()
        loss = self.get_loss(ip_pred, ip_true)
        loss.backward()
        # print(loss)
        self.optimizer.step()
        return loss

    def get_loss(self, ip_pred, ip_true):
        # sigmoid = nn.Sigmoid()
        # print(ip_pred.shape, ip_true.shape)
        # print(ip_pred)
        # print(ip_true)
        return self.loss_func(ip_pred, ip_true)