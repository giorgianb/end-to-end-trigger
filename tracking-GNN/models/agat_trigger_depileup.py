
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
import torch_geometric.nn as gnn
from torch_scatter import scatter_add
import pickle

import os

# Locals
from .utils import make_mlp

def load_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer
    return model

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
        self.network = make_mlp(input_dim*2+1,
                                [hidden_dim, hidden_dim, hidden_dim, 1],
                                # [hidden_dim, 1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        edge_inputs = torch.cat([x[start], x[end], e.unsqueeze(-1)], dim=1)
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
        layers = []
        prev_dim = input_dim
        hidden_activation = getattr(nn, hidden_activation)
        for i in range(3):
            layers.append((gnn.GATv2Conv(prev_dim, output_dim, edge_dim=1), 'x, edge_index, edge_attr -> x'))
            layers.append(hidden_activation())
            if layer_norm:
                layers.append(gnn.norm.LayerNorm(output_dim))
            prev_dim = output_dim

        layers.append((gnn.GATv2Conv(prev_dim, output_dim, edge_dim=1), 'x, edge_index, edge_attr -> x'))
        self.network = gnn.Sequential('x, edge_index, edge_attr', layers)


    def forward(self, x, e, edge_index):
        return self.network(x, edge_index, e)

class GNNGraphClassifier(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 hidden_activation='Tanh', layer_norm=True, add_n_hits=False, edge_model_path=None):
        super(GNNGraphClassifier, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
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
        if edge_model_path is not None:
            config_file = edge_model_path + '/config.pkl'
            config = pickle.load(open(config_file, 'rb'))
            model_config = config.get('model', {})
            model_config.pop('loss_func')
            model_config.pop('name')
            from models.agnn import GNNSegmentClassifier
            model = GNNSegmentClassifier(**model_config)

            checkpoint_dir = os.path.join(edge_model_path, 'checkpoints')
            checkpoint_file = sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('model_checkpoint')])
            checkpoint_file = checkpoint_file[-1]
            self.edge_model = load_checkpoint(checkpoint_file, model)




    def forward(self, inputs):
        """Apply forward pass of the model"""

        # Apply input network to get hidden representation
        # print(self.input_network.state_dict())
        x = self.input_network(inputs.x)
        temp = torch.cuda.FloatTensor(x.shape[0]).fill_(1)
        n_hits = scatter_add(temp, inputs.batch).view(-1, 1)

        e = self.edge_model(inputs)

        # Shortcut connect the inputs onto the hidden representation
        #x = torch.cat([x, inputs.x], dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):

            # Previous hidden state
            x0 = x


            # Apply node network
            x = self.node_network(x, e, inputs.edge_index)

            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, e, inputs.edge_index))

            # Shortcut connect the inputs onto the hidden representation
            #x = torch.cat([x, inputs.x], dim=-1)

            # Residual connection
            x = x + x0

        # Apply final graph network
        s = scatter_add(x, inputs.batch, dim=0)
        if self.add_n_hits:
            s = torch.cat([s, n_hits], axis=1)
        return self.graph_pred_mlp(s).squeeze(-1)
    
