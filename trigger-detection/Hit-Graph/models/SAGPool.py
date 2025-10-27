import torch
import torch_geometric.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from models.layers import SAGPool

class SAGPoolNet(torch.nn.Module):
    def __init__(self, is_hierarchical, num_features, nhid, num_classes, pooling_ratio, dropout_ratio, learning_rate=0.001, lr_scheduler_decrease_rate=0.95):
        super(SAGPoolNet, self).__init__()
        self.name = 'SAGPool'

        # self.args = args
        # self._num_features = num_features
        # self._nhid = nhid
        # self._num_classes = num_classes
        # self._pooling_ratio = pooling_ratio
        self._dropout_ratio = dropout_ratio
        self._is_hierarchical = is_hierarchical
        
        self.conv1 = GCNConv(num_features, nhid)
        self.pool1 = SAGPool(nhid, ratio=pooling_ratio)
        self.conv2 = GCNConv(nhid, nhid)
        self.pool2 = SAGPool(nhid, ratio=pooling_ratio)
        self.conv3 = GCNConv(nhid, nhid)
        self.pool3 = SAGPool(nhid, ratio=pooling_ratio)

        self.conv_global = GCNConv(nhid, nhid)
        self.pool_global = SAGPool(nhid*3, ratio=pooling_ratio)
        self.lin1_global = torch.nn.Linear(nhid*6, nhid)

        self.lin1 = torch.nn.Linear(nhid*2, nhid)
        self.lin2 = torch.nn.Linear(nhid, nhid//2)
        self.lin3 = torch.nn.Linear(nhid//2, num_classes)

        print("is_hierarchical: " + str(is_hierarchical))

        # optimizer init
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        # lr_scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)

    def forward(self, x, edge_index, batch, batch_size, edge_attr, is_e=True):

        if self._is_hierarchical:
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, edge_attr, batch, perm = self.pool1(x, edge_index, edge_attr, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, edge_attr, batch, perm = self.pool2(x, edge_index, edge_attr, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, edge_attr, batch, perm = self.pool3(x, edge_index, edge_attr, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            x = x1 + x2 + x3

            x = F.relu(self.lin1(x))
        else:
            x = F.relu(self.conv1(x, edge_index))
            x1 = x

            x = F.relu(self.conv2(x, edge_index))
            x2 = x
        
            x = F.relu(self.conv3(x, edge_index))
            x = torch.cat([x1, x2, x], dim = 1)

            x, edge_index, edge_attr, batch, perm = self.pool_global(x, edge_index, edge_attr, batch)
            x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.lin1_global(x))
        
        x = F.dropout(x, p=self._dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

    def train_model(self, ip_pred, ip_true):
        self.optimizer.zero_grad()
        ip_true = ip_true.long()
        loss = F.nll_loss(ip_pred, ip_true)
        # loss = self.get_loss(ip_pred, ip_true)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_loss(self, ip_pred, ip_true):
        ip_true = ip_true.long()
        loss = F.nll_loss(ip_pred, ip_true)
        return loss
    #     sigmoid = nn.Sigmoid()
    #     return self.loss_func(sigmoid(ip_pred), ip_true)