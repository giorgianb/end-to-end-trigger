import numpy as np
import os
import tqdm

PHI_SLOPE_MAX = 0.012193355583173944
Z0_MAX = 14.220353082111805
N_FILES = 20000

trigger_input_dir = '/ssd2/giorgian/hits-data-august-2022/trigger/1/'
trigger_file_names = sorted([os.path.join(trigger_input_dir, f) for f in os.listdir(trigger_input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
nontrigger_input_dir = '/ssd2/giorgian/hits-data-august-2022/nontrigger/0/'
nontrigger_file_names = sorted([os.path.join(nontrigger_input_dir, f) for f in os.listdir(nontrigger_input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])

trigger_output_dir = 'data/trigger/'
nontrigger_output_dir = 'data/nontrigger/'

import dataclasses

def load_file(filename):
    with np.load(filename) as f:
        return EventInfo(
                hit_cartesian=f['hit_cartesian'],
                hit_cylindrical=f['hit_cylindrical'],
                layer_id=f['layer_id'],
                n_pixels=f['n_pixels'],
                energy=f['energy'],
                momentum=f['momentum'],
                interaction_point=f['interaction_point'],
                trigger=f['trigger'],
                has_trigger_pair=f['has_trigger_pair'],
                track_origin=f['track_origin'],
                edge_index=f['edge_index'],
                edge_z0=f['edge_z0'],
                edge_phi_slope=f['edge_phi_slope'],
                phi_slope_max=f['phi_slope_max'],
                z0_max=f['z0_max'],
                trigger_node=f['trigger_node'],
                particle_id=f['particle_id'],
                particle_type=f['particle_type'],
                parent_particle_type=f['parent_particle_type'],
        )




@dataclasses.dataclass
class EventInfo:
    hit_cartesian: np.ndarray
    hit_cylindrical: np.ndarray
    layer_id: np.ndarray
    n_pixels: np.ndarray
    energy: np.ndarray
    momentum: np.ndarray
    interaction_point: np.ndarray
    trigger: np.ndarray
    has_trigger_pair: np.ndarray
    track_origin: np.ndarray
    edge_index: np.ndarray
    edge_z0: np.ndarray
    edge_phi_slope: np.ndarray
    phi_slope_max: float
    z0_max: float
    trigger_node: np.ndarray
    particle_id: np.ndarray
    particle_type: np.ndarray
    parent_particle_type: np.ndarray

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi


def build_edges(event_info, phi_slope_max, z0_max):
    r, phi, z  = event_info.hit_cylindrical.T
    layer_pairs = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (1, 3), (1,4), (2,4), (2,5), (3,5), (3,6), (4,6)]
    hit_ids = np.arange(len(event_info.hit_cylindrical))
    edge_candidates = []
    phi_slopes = []
    z0s = []
    layer_id = event_info.layer_id
    for (layer1, layer2) in layer_pairs:
        mask1 = layer_id == layer1
        mask2 = layer_id == layer2
        if np.sum(mask1) == 0 or np.sum(mask2) == 0:
            continue
        h1 = hit_ids[mask1]
        h2 = hit_ids[mask2]
        edges = np.stack(np.meshgrid(h1, h2, indexing='xy'), axis=-1).reshape(-1, 2)

        z1 = z[mask1]
        z2 = z[mask2]
        r1 = r[mask1]
        r2 = r[mask2]
        phi1 = phi[mask1]
        phi2 = phi[mask2]

        dphi = calc_dphi(phi2.reshape(-1, 1), phi1.reshape(1, -1))
        dr = r2.reshape(-1, 1) - r1.reshape(1, -1)
        dz = z2.reshape(-1, 1) - z1.reshape(1, -1)
        phi_slope = dphi / dr
        z0 = z1 - r1 * dz / dr
        good_seg_mask = (np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)
        good_seg_mask = good_seg_mask.reshape(-1)
        edge_candidates.append(edges[good_seg_mask])
        phi_slopes.append(phi_slope.reshape(-1)[good_seg_mask])
        z0s.append(z0.reshape(-1)[good_seg_mask])

    return np.concatenate(edge_candidates, axis=0).T, np.concatenate(phi_slopes), np.concatenate(z0s)





def load_graph(filename, cylindrical_features_scale, phi_slope_max, z0_max, use_intt):
    event_info = load_file(filename)

    if not use_intt:
        keep = event_info.layer_id <= 2
    else:
        keep = np.ones(event_info.layer_id.shape[0], dtype=bool)


    x = np.concatenate([
        event_info.hit_cylindrical/cylindrical_features_scale[None],
        event_info.n_pixels.reshape(-1, 1), 
        event_info.layer_id.reshape(-1, 1)
    ], axis=-1)



    edge_index = event_info.edge_index
    phi_slope = event_info.edge_phi_slope
    z0 = event_info.edge_z0
    edge_index = edge_index[:, (np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)]
    pid = event_info.particle_id
    keep_edge = keep[edge_index[0]] & keep[edge_index[1]]

    edge_index = edge_index[:, keep_edge]

    start, end = edge_index
    y = event_info.particle_id[start] == event_info.particle_id[end]
    trigger = event_info.trigger


    return x, edge_index, y, trigger


cylindrical_features_scale = np.array([3, 1, 3])
for file in tqdm.tqdm(trigger_file_names[:N_FILES]):
    x, edge_index, y, trigger = load_graph(file, cylindrical_features_scale, PHI_SLOPE_MAX, Z0_MAX, use_intt=True)
    basename = os.path.basename(file)
    np.savez(os.path.join(trigger_output_dir, basename), x=x, edge_index=edge_index, y=y, trigger=trigger)

for file in tqdm.tqdm(nontrigger_file_names[:N_FILES]):
    x, edge_index, y, trigger = load_graph(file, cylindrical_features_scale, PHI_SLOPE_MAX, Z0_MAX, use_intt=True)
    basename = os.path.basename(file)
    np.savez(os.path.join(nontrigger_output_dir, basename), x=x, edge_index=edge_index, y=y, trigger=trigger)
