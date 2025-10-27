

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
import math

import torch_geometric.nn as gnn
# Locals
from .utils import make_mlp

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > math.pi] -= 2*math.pi
    dphi[dphi < -math.pi] += 2*math.pi
    return dphi



class Id(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_intt, x_mvtx):
        return x_intt, x_mvtx

class BipartiteLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_activation, n_aggregators=8, aggregator_activation='potential', phi_slope_max=None, z0_max=None, apply_constraints=False):
        super().__init__()
        feature_dim = output_dim*2
        if aggregator_activation == 'potential':
            self.aggregator_activation = lambda x: torch.exp(-torch.abs(x))
        else:
            self.aggregator_activation = getattr(nn, aggregator_activation)()
        self.transform_in_intt = nn.Linear(input_dim, feature_dim)
        self.transform_in_mvtx = nn.Linear(input_dim, feature_dim)

        self.aggregator_score = nn.Linear(2*feature_dim, 1)
        self.transform_out_mvtx = nn.Sequential(
                nn.Linear(input_dim + feature_dim + 2*feature_dim + 2*feature_dim, output_dim),
                getattr(nn, hidden_activation)()
        )

        self.transform_out_intt = nn.Sequential(
                nn.Linear(input_dim + feature_dim + 2*feature_dim + 2*feature_dim, output_dim),
                getattr(nn, hidden_activation)()
        )

        self._feature_dim = feature_dim
        self._n_aggregators = n_aggregators
        self.phi_slope_max = phi_slope_max
        self.z0_max = z0_max
        self.apply_constraints = apply_constraints

    def forward(self, x_intt_0, x_mvtx_0, x_intt, x_mvtx, edge_index):
        xp_intt = self.transform_in_intt(x_intt)
        xp_mvtx = self.transform_in_mvtx(x_mvtx)
        if self.apply_constraints:
            start, end = edge_index
            # Calculate phi_slope, z0 and filter them out
            r2, phi2, z2 = x_intt_0[start][:, :3].T
            r1, phi1, z1 = x_mvtx_0[end][:, :3].T
            # Undo the cylindrical feature scaling we do
            r2 *= 3
            r1 *= 3
            z2 *= 3
            z1 *= 3
            dphi = calc_dphi(phi1, phi2)
            dr = r2 - r1
            dz = z2 - z1
            phi_slope = dphi / dr
            z0 = z1 - r1 * dz / dr

            good_seg_mask = (torch.abs(phi_slope) <= self.phi_slope_max) & (torch.abs(z0) <= self.z0_max)
            edge_index = edge_index[:, good_seg_mask]



        start, end = edge_index
        xp = torch.cat([xp_intt[start], xp_mvtx[end]], dim=-1)

        attention_score = self.aggregator_activation(self.aggregator_score(xp))
        edges = xp * attention_score

        # We want aggregators with the same node to add
        out_mean_intt = torch.zeros(xp_intt.shape[0], edges.shape[1], device=xp_intt.device)
        mean_pooled_intt = scatter_mean(edges, start, dim=0, out=out_mean_intt)

        out_max_intt = torch.zeros(xp_intt.shape[0], edges.shape[1], device=xp_intt.device)
        max_pooled_intt = scatter_max(edges, start, dim=0, out=out_max_intt)[0]
        # Now we have (a, f, b)
        out_mean_mvtx = torch.zeros(xp_mvtx.shape[0], edges.shape[1], device=xp_mvtx.device)
        mean_pooled_mvtx = scatter_mean(edges, end, dim=0, out=out_mean_mvtx)
        out_max_mvtx = torch.zeros(xp_mvtx.shape[0], edges.shape[1], device=xp_mvtx.device)
        max_pooled_mvtx = scatter_max(edges, end, dim=0, out=out_max_mvtx)[0]

        aggregators_intt = torch.cat([mean_pooled_intt, max_pooled_intt], dim=-1)
        H_intt = torch.cat([x_intt, xp_intt, aggregators_intt], dim=-1)
        aggregators_mvtx = torch.cat([mean_pooled_mvtx, max_pooled_mvtx], dim=-1)
        H_mvtx = torch.cat([x_mvtx, xp_mvtx, aggregators_mvtx], dim=-1)

        h_intt = self.transform_out_intt(H_intt)
        h_mvtx = self.transform_out_mvtx(H_mvtx)
        return h_intt, h_mvtx


class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, hidden_activation='ReLU',
            layer_norm=True, n_layers=3, phi_slope_max=None, z0_max=None, apply_constraints=False):
        super(NodeNetwork, self).__init__()
        layers = [
                (gnn.GATv2Conv(input_dim, output_dim), 'x_intt, edge_index_intt -> x_intt'),
                (BipartiteLayer(output_dim, output_dim, hidden_activation, phi_slope_max=phi_slope_max, z0_max=z0_max, apply_constraints=apply_constraints), 'x_intt_0, x_mvtx_0, x_intt, x_mvtx, edge_index -> x_intt, x_mvtx')
        ]

        if layer_norm:
            layers.append(gnn.norm.LayerNorm(output_dim))


        for i in range(n_layers-1):
            layers.append(
                (gnn.GATv2Conv(output_dim, output_dim), 'x_intt, edge_index_intt -> x_intt'))
            layers.append(
                (BipartiteLayer(output_dim, output_dim, hidden_activation, phi_slope_max=phi_slope_max, z0_max=z0_max, apply_constraints=apply_constraints), 'x_intt_0, x_mvtx_0, x_intt, x_mvtx, edge_index -> x_intt, x_mvtx')
            )

            if layer_norm:
                layers.append(gnn.norm.LayerNorm(output_dim))

        layers.append(
            (gnn.GATv2Conv(output_dim, output_dim), 'x_intt, edge_index_intt -> x_intt'))

        if layer_norm:
            layers.append(gnn.norm.LayerNorm(output_dim))


        layers.append((Id(), 'x_intt, x_mvtx -> x_intt, x_mvtx'))
        self.network = gnn.Sequential('x_intt_0, x_mvtx_0, x_intt, x_mvtx, edge_index, edge_index_intt', layers)

    def forward(self, x_intt_0, x_mvtx_0, x_intt, x_mvtx, edge_index, edge_index_intt):
        ret = self.network(x_intt_0, x_mvtx_0, x_intt, x_mvtx, edge_index, edge_index_intt)

        return ret

class GNNSegmentClassifier(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, intt_input_dim=3, mvtx_input_dim=3, hidden_dim=8, n_graph_iters=3, n_layers=3, track_dim=15, hidden_activation='ReLU', layer_norm=True, phi_slope_max=None, z0_max=None, apply_constraints=False):
        super().__init__()
        self.apply_constraints = apply_constraints
        self.phi_slope_max = phi_slope_max
        self.z0_max = z0_max
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network_intt = make_mlp(intt_input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)

        self.input_network_mvtx = make_mlp(mvtx_input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)

        # Setup the node layers
        self.node_network = NodeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm, n_layers=n_layers, phi_slope_max=phi_slope_max, z0_max=z0_max, apply_constraints=apply_constraints)

        self.edge_network = make_mlp(hidden_dim*2,
                                [hidden_dim, hidden_dim, hidden_dim, 1],
                                # [hidden_dim, 1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)
        self.track_pred = make_mlp(hidden_dim, [hidden_dim]*3 + [track_dim],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)



        

    def forward(self, inputs):
        """Apply forward pass of the model"""
        x_intt = inputs.x_intt
        x_mvtx = inputs.x_mvtx
        edge_index = inputs.edge_index
        edge_index_intt = inputs.edge_index_intt

        # Make a copy
        x_intt_0 = 1.0*x_intt + 0.0
        x_mvtx_0 = 1.0*x_mvtx + 0.0
        # Apply input network to get hidden representation
        # print(self.input_network.state_dict())
        x_intt = self.input_network_intt(x_intt)
        x_mvtx = self.input_network_mvtx(x_mvtx)

        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):

            # Apply node network
            x_intt, x_mvtx = self.node_network(x_intt_0, x_mvtx_0, x_intt, x_mvtx, edge_index, edge_index_intt)

        return self.track_pred(x_intt)

