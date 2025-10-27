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
from torch_scatter import scatter_add, scatter_mean

# Locals
from .utils import make_mlp

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
        self.network = make_mlp(input_dim, [output_dim]*4,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x):
        return self.network(x)

class GNNGraphClassifier(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 hidden_activation='Tanh', layer_norm=True, add_n_hits=False):
        super(GNNGraphClassifier, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        
        if add_n_hits:
            self.graph_pred_mlp = make_mlp(hidden_dim+1, [hidden_dim]*3 + [1],
                                    hidden_activation=hidden_activation,
                                    output_activation=None,
                                    layer_norm=layer_norm)
        else:
            self.graph_pred_mlp = make_mlp(hidden_dim, [hidden_dim]*3 + [1],
                                    hidden_activation=hidden_activation,
                                    output_activation=None,
                                    layer_norm=layer_norm)
        self.add_n_hits = add_n_hits

    def forward(self, inputs):
        """Apply forward pass of the model"""

        # Apply input network to get hidden representation
        # print(self.input_network.state_dict())
        x = self.input_network(inputs.x)
        temp = torch.cuda.FloatTensor(x.shape[0]).fill_(1)
        n_hits = scatter_add(temp, inputs.batch).view(-1, 1)

        # Shortcut connect the inputs onto the hidden representation
        #x = torch.cat([x, inputs.x], dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):

            # Previous hidden state
            x0 = x

            # Apply node network
            x = self.node_network(x)

            # Residual connection
            x = x + x0

        # Apply final graph network
        s = scatter_mean(x, inputs.batch, dim=0)
        if self.add_n_hits:
            s = torch.cat([s, n_hits], axis=1)
        return self.graph_pred_mlp(s).squeeze(-1)
    
