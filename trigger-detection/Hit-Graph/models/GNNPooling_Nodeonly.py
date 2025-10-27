"""
This module implements the PyTorch modules that define the sparse
message-passing graph neural networks for segment classification.
In particular, this implementation utilizes the pytorch_geometric
and supporting libraries:
https://github.com/rusty1s/pytorch_geometric
"""

# Externals
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.nn import global_max_pool as gmp

# Locals
from .utils import make_mlp

class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation='Tanh',
                 layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2,
                                [hidden_dim, hidden_dim, hidden_dim, 1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        x1, x2 = x[start], x[end]
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.network(edge_inputs).squeeze(-1)

class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, hidden_activation='Tanh',
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*3, [output_dim]*4,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
        mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([mi, mo, x], dim=1)
        return self.network(node_inputs)

class GNN(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 hidden_activation='Tanh', layer_norm=True, learning_rate=0.001, lr_scheduler_decrease_rate=0.9):
        super(GNN, self).__init__()

        self.name = 'GNN'
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        # self.edge_network = EdgeNetwork(hidden_dim, hidden_dim,
        #                                 hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)

        self.output_network = make_mlp(hidden_dim, [hidden_dim, 1],
                                      hidden_activation=hidden_activation,
                                      output_activation=None,
                                      layer_norm=layer_norm)
        # optimizer init
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        # lr_scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)
        loss_func='binary_cross_entropy_with_logits'
        self.loss_func = getattr(nn.functional, loss_func)

    def forward(self, x, edge_index, batch, batch_size, e=None, is_e=False):
        """Apply forward pass of the model"""

        # Apply input network to get hidden representation
        # print(self.input_network.state_dict())
        x = self.input_network(x)
        e = e.squeeze(-1)

        # Shortcut connect the inputs onto the hidden representation
        #x = torch.cat([x, inputs.x], dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):

            # Previous hidden state
            x0 = x

            # Apply edge network
            # e = torch.sigmoid(self.edge_network(x, edge_index))

            # Apply node network
            x = self.node_network(x, e, edge_index)

            # Shortcut connect the inputs onto the hidden representation
            #x = torch.cat([x, inputs.x], dim=-1)

            # Residual connection
            x = x + x0

        # Apply final edge network
        # edge_output = self.edge_network(x, edge_index)

        node_output = x

        # summary = scatter_add(node_output, batch, dim=0, dim_size=batch_size) /(x.shape[0]/batch_size)
        summary = gmp(node_output, batch)

        return self.output_network(summary)

    def train_model(self, ip_pred, ip_true):
        self.optimizer.zero_grad()
        loss = self.get_loss(ip_pred, ip_true)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_loss(self, ip_pred, ip_true):
        # sigmoid = nn.Sigmoid()
        # return self.loss_func(sigmoid(ip_pred), ip_true)
        return self.loss_func(ip_pred, ip_true)