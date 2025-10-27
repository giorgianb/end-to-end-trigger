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
from torch_scatter import scatter_add, scatter_mean, scatter_max

import torch_geometric.nn as gnn
# Locals
from .utils import make_mlp

class BipartiteLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_activation, n_aggregators=8, aggregator_activation='potential'):
        super().__init__()
        feature_dim = output_dim*2
        if aggregator_activation == 'potential':
            self.aggregator_activation = lambda x: torch.exp(-torch.abs(x))
        else:
            self.aggregator_activation = getattr(nn, aggregator_activation)()
        self.transform_in = nn.Linear(input_dim, feature_dim)

        self.aggregator_score = nn.Linear(feature_dim, n_aggregators)
        self.transform_out = nn.Sequential(
                nn.Linear(2*feature_dim*n_aggregators + input_dim + feature_dim, output_dim),
                getattr(nn, hidden_activation)()
        )
        self._feature_dim = feature_dim
        self._n_aggregators = n_aggregators

    def forward(self, x, batch):
        xp = self.transform_in(x)
        attention_score = self.aggregator_activation(self.aggregator_score(xp))

        # Weight the edges
        # aggregator, feature, node
        edges = torch.einsum('nf,na->afn', xp, attention_score)

        # We want aggregators with the same node to add
        mean_pooled = scatter_mean(edges, batch, dim=-1)
        max_pooled = scatter_max(edges, batch, dim=-1)[0]
        # Now we have (a, f, b)

        # (a, f, b) -> (b, f, a) -> (b, a, f)
        mean_pooled = mean_pooled.transpose(2, 0).transpose(1, 2)
        max_pooled = max_pooled.transpose(2, 0).transpose(1, 2)

        aggregators = torch.cat([mean_pooled, max_pooled], dim=-1)
        aggregators = aggregators.reshape(aggregators.shape[0], -1)

        # Now we have (b, f)

        # Now we need to scatter_add: if a node belongs to batch i, add aggregators[i] to it

        aggregators = aggregators[batch]
        H = torch.cat([x, xp, aggregators], dim=-1)

        h = self.transform_out(H)
        return h


class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, hidden_activation='Tanh',
            layer_norm=True, aggregator_activation='potential', n_aggregators=8):
        super(NodeNetwork, self).__init__()
        layers = []
        for i in range(3):
            layers.append(
                (BipartiteLayer(input_dim, output_dim, hidden_activation, n_aggregators=n_aggregators, aggregator_activation=aggregator_activation), 'x, batch -> x')
            )

            if layer_norm:
                layers.append(gnn.norm.LayerNorm(output_dim))

        self.network = gnn.Sequential('x, batch', layers)

    def forward(self, x, batch):
        return self.network(x, batch)

class GNNGraphClassifier(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
            hidden_activation='Tanh', layer_norm=True, add_n_hits=False, aggregator_activation='potential'):
        super(GNNGraphClassifier, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm, aggregator_activation=aggregator_activation)
        
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
            x = self.node_network(x, inputs.batch)

        # Apply final graph network
        s = scatter_mean(x, inputs.batch, dim=0)
        if self.add_n_hits:
            s = torch.cat([s, n_hits], axis=1)
        return self.graph_pred_mlp(s).squeeze(-1)
    
