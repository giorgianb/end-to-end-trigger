
"""Dataset specification for hit graphs using pytorch_geometric formuation"""
""" "Bipartite" formulation. INTT nodes are grouped together based on layer 4."""
"""This one is compatible with the older models; does not create a seperate x_mvtx and x_intt"""

# System imports
import os

# External imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import torch_geometric
import random

from collections import namedtuple
import dataclasses

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
    active_node: np.ndarray

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi



def group_intt(event_info):
    hit_cylindrical = event_info.hit_cylindrical
    hit_cartesian = event_info.hit_cartesian
    layer_id = event_info.layer_id
    n_pixels = event_info.n_pixels
    particle_id = event_info.particle_id
    trigger_node = event_info.trigger_node


    intt_hits_mask = layer_id >= 3
    hc = hit_cylindrical[intt_hits_mask]
    hcart = hit_cartesian[intt_hits_mask]
    pid = particle_id[intt_hits_mask]
    trg = trigger_node[intt_hits_mask]

    li = layer_id[intt_hits_mask]
    npx = n_pixels[intt_hits_mask]


    h1 = hcart[li <= 4]
    h2 = hcart[li > 4]
    assert len(h1) > 0 or len(h2) > 0
    if len(h2) > 0:
        diff = h1.reshape(-1, 1, 3) - h2.reshape(1, -1, 3)
        # first entry is h1, second entry is h2
        dist = np.linalg.norm(diff, axis=-1)
        closest = np.argmin(dist, axis=-1)

        hc1 = hc[li <= 4]
        li1 = li[li <= 4]
        npx1 = npx[li <= 4]
        pid1 = pid[li <= 4]
        trg1 = trg[li <= 4]
        hc2 = hc[li > 4]
        li2 = li[li > 4]
        npx2 = npx[li > 4]

        h_intt = np.concatenate([hc1, hc2[closest]], axis=-1)
        li_intt = np.stack([li1, li2[closest]], axis=-1)
        npx_intt = np.stack([npx1, npx2[closest]], axis=-1)
    else:
        hc1 = hc[li <= 4]
        li1 = li[li <= 4]
        npx1 = npx[li <= 4]
        pid1 = pid[li <= 4]
        trg1 = trg[li <= 4]
 
        h_intt = np.concatenate([hc1, np.zeros_like(hc1)], axis=-1)
        li_intt = np.stack([li1, np.zeros_like(li1)], axis=-1)
        npx_intt = np.stack([npx1, np.zeros_like(npx1)], axis=-1)

    return h_intt, npx_intt, li_intt, pid1, trg1, intt_hits_mask


def build_edges(hit_cylindrical_intt, hit_cylindrical_mvtx, phi_slope_max, z0_max):
    r1, phi1, z1  = hit_cylindrical_intt[:, :3].T
    r2, phi2, z2 = hit_cylindrical_mvtx.T
    # "Bipartite" layer pairs
    h1 = np.arange(len(hit_cylindrical_intt))
    h2 = np.arange(len(hit_cylindrical_mvtx))
    edge_candidates = []
    phi_slopes = []
    z0s = []
    edges = np.stack(np.meshgrid(h1, h2, indexing='xy'), axis=-1).reshape(-1, 2)

    dphi = calc_dphi(phi2.reshape(-1, 1), phi1.reshape(1, -1))
    dr = r2.reshape(-1, 1) - r1.reshape(1, -1)
    dz = z2.reshape(-1, 1) - z1.reshape(1, -1)
    phi_slope = dphi / dr
    z0 = z1 - r1 * dz / dr
    good_seg_mask = (np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)
    good_seg_mask = good_seg_mask.reshape(-1)
    edge_candidates = edges[good_seg_mask]
    phi_slopes = phi_slope.reshape(-1)
    z0s = z0.reshape(-1)


    return edge_candidates.T, phi_slopes, z0s

def has_intt_hits(filename):
    with np.load(filename, allow_pickle=True) as f:
        return np.any(f['layer_id'] >= 3)


def load_file(filename):
    with np.load(filename, allow_pickle=True) as f:
        if 'parent_particle_type' in f.keys():
            parent_particle_type = f['parent_particle_type']
        else:
            parent_particle_type = np.ones(f['hit_cylindrical'].shape[0])
        if has_trigger_pair in f.keys():
            has_trigger_pair = f['has_trigger_pair']
        else:
            has_trigger_pair = np.array(np.nan)
        e =  EventInfo(
                hit_cartesian=f['hit_cartesian'],
                hit_cylindrical=f['hit_cylindrical'],
                layer_id=f['layer_id'],
                n_pixels=f['n_pixels'],
                energy=f['energy'],
                momentum=f['momentum'],
                interaction_point=f['interaction_point'],
                trigger=f['trigger'],
                has_trigger_pair=has_trigger_pair,
                track_origin=f['track_origin'] if f['track_origin'].dtype != np.dtype('O') else np.stack(f['track_origin'], axis=0),
                edge_index=f['edge_index'],
                edge_z0=f['edge_z0'],
                edge_phi_slope=f['edge_phi_slope'],
                phi_slope_max=f['phi_slope_max'],
                z0_max=f['z0_max'],
                trigger_node=f['trigger_node'],
                particle_id=f['particle_id'],
                particle_type=f['particle_type'],
                parent_particle_type=parent_particle_type,
                active_node=np.ones(f['hit_cartesian'].shape[0], dtype=int)
        )
    return e


def load_graph(filename, cylindrical_features_scale, phi_slope_max, z0_max, use_intt, construct_edges=True, drop_l1=False, drop_l2=False, drop_l3=False, add_global_node=False):
    assert not (drop_l1 or drop_l2 or drop_l3 or add_global_node), 'Not implemented'
    event_info = load_file(filename)

    if not use_intt:
        keep = event_info.layer_id <= 2
    else:
        keep = np.ones(event_info.layer_id.shape[0]).astype(bool)

    x = np.concatenate([
        event_info.hit_cylindrical/cylindrical_features_scale[None],
        event_info.n_pixels.reshape(-1, 1), 
        event_info.layer_id.reshape(-1, 1)
    ], axis=-1)[keep]



    edge_index = event_info.edge_index
    phi_slope = event_info.edge_phi_slope
    z0 = event_info.edge_z0
    edge_index = edge_index[:, (np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)]
    pid = event_info.particle_id
    keep_edge = keep[edge_index[0]] & keep[edge_index[1]]

    edge_index = edge_index[:, keep_edge]

    y = pid[edge_index[0]] == pid[edge_index[1]]

    event_info.edge_index = edge_index

    event_info.edge_phi_slope = (phi_slope[(np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)])[keep_edge]
    event_info.edge_z0 = (z0[(np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)])[keep_edge]
    event_info.phi_slope_max = phi_slope_max
    event_info.z0_max = z0_max


    return x, edge_index, y, event_info

def intt_mask(og_event, pileup_event, phi_slope_max, z0_max):
    intt_hits = og_event.hit_cylindrical[og_event.layer_id >= 3]
    mvtx_hits = pileup_event.hit_cylindrical
    r1, phi1, z1 = intt_hits.T
    r2, phi2, z2 = mvtx_hits.T

    dphi = calc_dphi(phi2.reshape(-1, 1), phi1.reshape(1, -1))
    dr = r2.reshape(-1, 1) - r1.reshape(1, -1)
    dz = z2.reshape(-1, 1) - z1.reshape(1, -1)
    dr[dr == 0] = 1
    phi_slope = dphi / dr
    z0 = z1 - r1 * dz / dr
    good_seg_mask = (np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)
    return np.any(good_seg_mask, axis=-1)


    # so now we know what edges from intt_hits to pileup_events are good
    # The issue is, we now need to keep all 

 



def multi_load_graph(intt_filename, filenames, noise_filenames, cylindrical_features_scale, phi_slope_max, z0_max, use_intt, n_noise_intt=3, construct_edges=True, drop_l1=False, drop_l2=False, drop_l3=False, intt_filter=False, add_global_node=False):
    event_info_list = [load_file(intt_filename)] + [load_file(filename) for filename in filenames]
    noise_event_info_list = [load_file(filename) for filename in noise_filenames]
    max_pid = 0
    start_index = 1 if use_intt else 0
    for i, event_info in enumerate(event_info_list):
        # Node mask
        if i == 0:
            if use_intt:
                mask = np.ones_like(event_info.layer_id).astype(bool)
            else:
                mask = event_info.layer_id < 3
        else:
            mask = event_info.layer_id < 3

        if drop_l1:
            mask &= (event_info.layer_id != 0)
        if drop_l2:
            mask &= (event_info.layer_id != 1)
        if drop_l3:
            mask &= (event_info.layer_id != 2)

        if intt_filter:
            mask &= intt_mask(event_info_list[0], event_info, phi_slope_max, z0_max)

        event_info.hit_cartesian = event_info.hit_cartesian[mask]
        event_info.hit_cylindrical = event_info.hit_cylindrical[mask]
        event_info.layer_id = event_info.layer_id[mask]
        event_info.n_pixels = event_info.n_pixels[mask]
        event_info.momentum = event_info.momentum[mask]
        event_info.energy = event_info.energy[mask]
        event_info.particle_id = event_info.particle_id[mask] + max_pid
        event_info.track_origin = event_info.track_origin[mask]
        max_pid = np.max(event_info.particle_id, initial=0)
        event_info.particle_type = event_info.particle_type[mask]
        event_info.parent_particle_type = event_info.parent_particle_type[mask]
        event_info.trigger_node = event_info.trigger_node[mask]
        event_info.active_node = event_info.active_node[mask]*(i == start_index and not use_intt)

    for i, event_info in enumerate(noise_event_info_list):
        # Node mask
        mask = np.ones(event_info.hit_cartesian.shape[0], dtype=bool)
        if drop_l1:
            mask &= (event_info.layer_id != 0)
        if drop_l2:
            mask &= (event_info.layer_id != 1)
        if drop_l3:
            mask &= (event_info.layer_id != 2)


        event_info.hit_cartesian = event_info.hit_cartesian[mask]
        event_info.hit_cylindrical = event_info.hit_cylindrical[mask]
        event_info.layer_id = event_info.layer_id[mask]
        event_info.n_pixels = np.floor(np.random.exponential(size=mask.shape)+1).astype(np.int32)[mask]
        event_info.momentum = np.nan*np.ones_like(event_info.momentum[mask])
        event_info.energy = np.nan*np.ones_like(event_info.energy[mask])
        event_info.particle_id = np.nan*np.ones_like(event_info.particle_id[mask])
        event_info.track_origin = np.nan*np.ones_like(event_info.track_origin[mask])
        event_info.particle_type = np.nan*np.ones_like(event_info.particle_type[mask])
        event_info.parent_particle_type = np.nan*np.ones_like(event_info.parent_particle_type[mask])
        event_info.trigger_node = np.zeros_like(event_info.trigger_node[mask])
        event_info.active_node = np.zeros_like(event_info.active_node[mask])
        

    noise_event_info = EventInfo(
            hit_cartesian=np.concatenate([event_info.hit_cartesian for event_info in noise_event_info_list], axis=0),
            hit_cylindrical=np.concatenate([event_info.hit_cylindrical for event_info in noise_event_info_list], axis=0),
            track_origin=np.concatenate([event_info.track_origin for event_info in noise_event_info_list], axis=0),
            layer_id=np.concatenate([event_info.layer_id for event_info in noise_event_info_list], axis=0),
            n_pixels=np.concatenate([event_info.n_pixels for event_info in noise_event_info_list], axis=0),
            energy=np.concatenate([event_info.energy for event_info in noise_event_info_list], axis=0),
            momentum=np.concatenate([event_info.momentum for event_info in noise_event_info_list], axis=0),
            interaction_point=noise_event_info_list[0].interaction_point,
            trigger=False,
            has_trigger_pair=noise_event_info_list[0].has_trigger_pair,
            particle_id=np.concatenate([event_info.particle_id for event_info in noise_event_info_list], axis=0),
            particle_type=np.concatenate([event_info.particle_type for event_info in noise_event_info_list], axis=0),
            parent_particle_type=np.concatenate([event_info.parent_particle_type for event_info in noise_event_info_list], axis=0),
            trigger_node=np.concatenate([event_info.trigger_node for event_info in noise_event_info_list], axis=0),
            active_node=np.concatenate([event_info.active_node for event_info in noise_event_info_list], axis=0),
            edge_index=None,
            edge_z0=None,
            edge_phi_slope=None,
            phi_slope_max=phi_slope_max,
            z0_max=z0_max
        )


    
    n_mvtx_noise = np.random.binomial(n=1572864*len(event_info_list), p=1e-6)
    n_noise_intt = int(np.floor(np.random.exponential()+n_noise_intt))
    mvtx_noise_hits = np.where(noise_event_info.layer_id < 3)[0]
    intt_noise_hits = np.where(noise_event_info.layer_id >= 3)[0]
    np.random.shuffle(mvtx_noise_hits)
    np.random.shuffle(intt_noise_hits)
    mvtx_noise_hits = mvtx_noise_hits[:n_mvtx_noise]
    intt_noise_hits = intt_noise_hits[:n_noise_intt]

    mask = np.zeros(noise_event_info.hit_cartesian.shape[0], dtype=bool)
    mask[mvtx_noise_hits] = 1
    if use_intt:
        mask[intt_noise_hits] = 1

    noise_event_info.hit_cartesian = noise_event_info.hit_cartesian[mask]
    noise_event_info.hit_cylindrical = noise_event_info.hit_cylindrical[mask]
    noise_event_info.track_origin = noise_event_info.track_origin[mask]
    noise_event_info.layer_id = noise_event_info.layer_id[mask]
    noise_event_info.n_pixels = noise_event_info.n_pixels[mask]
    noise_event_info.energy = noise_event_info.energy[mask]
    noise_event_info.momentum = noise_event_info.momentum[mask]
    noise_event_info.particle_id = noise_event_info.particle_id[mask]
    noise_event_info.parent_particle_type = noise_event_info.parent_particle_type[mask]
    noise_event_info.trigger_node = noise_event_info.trigger_node[mask]
    noise_event_info.active_node = noise_event_info.active_node[mask]

    event_info_list.append(noise_event_info)
    trigger = event_info_list[0].trigger if use_intt else np.array(any(ev.trigger for ev in event_info_list))

    files = [intt_filename] + filenames
    for i, event_info in enumerate(event_info_list):
        if event_info.track_origin.shape[-1] != 3:
            print(f'{files[i]=} {event_info.track_origin.shape=}')

    event_info = EventInfo(
            hit_cartesian=np.concatenate([event_info.hit_cartesian for event_info in event_info_list], axis=0),
            hit_cylindrical=np.concatenate([event_info.hit_cylindrical for event_info in event_info_list], axis=0),
            track_origin=np.concatenate([event_info.track_origin for event_info in event_info_list], axis=0),
            layer_id=np.concatenate([event_info.layer_id for event_info in event_info_list], axis=0),
            n_pixels=np.concatenate([event_info.n_pixels for event_info in event_info_list], axis=0),
            energy=np.concatenate([event_info.energy for event_info in event_info_list], axis=0),
            momentum=np.concatenate([event_info.momentum for event_info in event_info_list], axis=0),
            interaction_point=event_info_list[0].interaction_point,
            trigger=trigger,
            has_trigger_pair=event_info_list[0].has_trigger_pair,
            particle_id=np.concatenate([event_info.particle_id for event_info in event_info_list], axis=0),
            particle_type=np.concatenate([event_info.particle_type for event_info in event_info_list], axis=0),
            parent_particle_type=np.concatenate([event_info.parent_particle_type for event_info in event_info_list], axis=0),
            trigger_node=np.concatenate([event_info.trigger_node for event_info in event_info_list], axis=0),
            active_node=np.concatenate([event_info.active_node for event_info in event_info_list]),
            edge_index=None,
            edge_z0=None,
            edge_phi_slope=None,
            phi_slope_max=phi_slope_max,
            z0_max=z0_max
        )


    hit_cylindrical_intt, n_pixels_intt, layer_id_intt, particle_id_intt, trigger_node_intt, mask = group_intt(event_info)
    hit_cylindrical_mvtx, n_pixels_mvtx, layer_id_mvtx, particle_id_mvtx, trigger_node_mvtx = event_info.hit_cylindrical[~mask], event_info.n_pixels[~mask], event_info.layer_id[~mask], event_info.particle_id[~mask], event_info.trigger_node[~mask]
    x_mvtx = np.concatenate([
        event_info.hit_cylindrical[~mask]/cylindrical_features_scale[None],
        event_info.n_pixels.reshape(-1, 1)[~mask],
        event_info.layer_id.reshape(-1, 1)[~mask],
    ], axis=-1)

    x_mvtx_pad = np.zeros_like(x_mvtx)
    x_mvtx = np.concatenate([x_mvtx, x_mvtx_pad], axis=-1)

    x_intt = np.concatenate([
        hit_cylindrical_intt,
        n_pixels_intt,
        layer_id_intt
    ], axis=-1)

    trigger_node = np.concatenate([trigger_node_intt, trigger_node_mvtx], axis=0)


    x = np.concatenate([x_intt, x_mvtx], axis=0)
    if construct_edges:
        edge_index, phi_slope, z0 = build_edges(hit_cylindrical_intt, hit_cylindrical_mvtx, phi_slope_max, z0_max)
        start, end = edge_index
        y = particle_id_intt[start] == particle_id_mvtx[end]
        edge_index[1] += len(x_intt)
    else:
        edge_index = np.zeros((2, 0), dtype=int)
        phi_slope = np.zeros(0)
        z0 = np.zeros(0)

        start, end = edge_index
        y = particle_id_intt[start] == particle_id_mvtx[end]


    if add_global_node:
        global_node = np.zeros((1, x.shape[1]))
        x = np.concatenate([x, global_node], axis=0)
        if construct_edges:
            global_edge_index = np.zeros((2, x.shape[0]-1), dtype=int)
            global_edge_index[0] = np.arange(x.shape[0]-1)
            edge_index = np.concatenate([edge_index, global_edge_index], axis=1)

    #event_info.edge_index = edge_index
    #event_info.edge_phi_slope = phi_slope
    #event_info.edge_z0 = z0

    

    return x, edge_index, y, event_info, trigger_node


class HitGraphDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir=None, filelist=None, n_samples=None, real_weight=1.0, n_folders=1, input_dir2=None, phi_slope_max=0.03, z0_max=200, n_mix=1, use_intt=False, cylindrical_features_scale=(3, 1, 3), load_full_event=False, load_all=False, construct_edges=True, drop_l1=False, drop_l2=False, drop_l3=False, intt_filter=False, add_global_node=False, ramp_up_nmix=False, ramp_rate=1, random_n_mix=False, min_random_n_mix=1, trigger_edge_weight=1):
        self.epoch = 0
        self.ramp_up_nmix = ramp_up_nmix
        self.ramp_rate = ramp_rate
        self.random_n_mix = random_n_mix
        self.min_random_n_mix = min_random_n_mix
        self.load_full_event = load_full_event
        self.cylindrical_features_scale = np.array(cylindrical_features_scale)
        self.construct_edges = construct_edges
        self.drop_l1 = drop_l1
        self.drop_l2 = drop_l2
        self.drop_l3 = drop_l3
        self.intt_filter = intt_filter
        if filelist is not None:
            self.metadata = pd.read_csv(os.path.expandvars(filelist))
            filenames = self.metadata.file.values
        elif input_dir is not None:
            input_dir = os.path.expandvars(input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
            n_samples_1 = min(len(filenames), n_samples)
            self.filenames_1 = random.sample(filenames, n_samples_1)
            self.f1_type = 'event1' in self.filenames_1[0]
        else:
            raise Exception('Must provide either input_dir or filelist to HitGraphDataset')
        if load_all:
            n_samples = len(filenames)

        self.filenames = filenames if n_samples is None else filenames[:n_samples]
        if n_folders == 2:
            filenames = sorted([os.path.join(input_dir2, f) for f in os.listdir(input_dir2)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
            #self.filenames_2 = filenames[:n_samples]
            n_samples_2 = min(len(filenames), n_samples)
            self.filenames_2 = random.sample(filenames, n_samples_1)
 
            self.filenames_2 = random.sample(filenames, n_samples)
            self.f2_type = 'event1' in self.filenames_2[0]
            if load_all:
                n_samples = len(filenames)

            self.filenames += filenames[:n_samples]
        self.real_weight = real_weight
        self.fake_weight = 1 #real_weight / (2 * real_weight - 1)
        self.phi_slope_max = phi_slope_max
        self.z0_max = z0_max
        self.n_mix = n_mix
        self.use_intt = use_intt
        self.n_noise_files = 2
        self.add_global_node = add_global_node
        self.trigger_edge_weight = trigger_edge_weight

    def __getitem__(self, index):
        event_file_name = self.filenames[index]

        if self.ramp_up_nmix:
            n_mix = min(self.n_mix, (self.epoch + 1) * self.ramp_rate) 
        elif self.random_n_mix:
            n_mix = random.randint(self.min_random_n_mix, self.n_mix)
        else:
            n_mix = self.n_mix

        if self.n_mix == 1:
            x, edge_index, y, event_info, trigger_node = load_graph(self.filenames[index], self.cylindrical_features_scale, self.phi_slope_max, self.z0_max, use_intt=self.use_intt, construct_edges=self.construct_edges, drop_l1=self.drop_l1, drop_l2=self.drop_l2, drop_l3=self.drop_l3, add_global_node=self.add_global_node)
        elif self.n_mix >= 2 and self.use_intt:

            event_file_name = self.filenames[index]
            while not has_intt_hits(event_file_name):
                event_file_name = random.sample(self.filenames, 1)[0]

            files = random.sample(self.filenames, n_mix)
            noise_filenames = random.sample(self.filenames, self.n_noise_files)
            if event_file_name in files:
                files.remove(event_file_name)
            else:
                files.pop()

            while set(noise_filenames) & set(files):
                noise_filenames = random.sample(self.filenames, self.n_noise_files)
            x, edge_index, y, event_info, trigger_node = multi_load_graph(event_file_name, files, noise_filenames, self.cylindrical_features_scale, self.phi_slope_max, self.z0_max, use_intt=self.use_intt, construct_edges=self.construct_edges, drop_l1=self.drop_l1, drop_l2=self.drop_l2, drop_l3=self.drop_l3, intt_filter=self.intt_filter, add_global_node=self.add_global_node)
        elif self.n_mix >= 2 and not self.use_intt:
            if np.random.uniform() >= 0.5:
                # Only non-trigger files
                files = random.sample(self.filenames_2, n_mix)

                noise_filenames = random.sample(self.filenames, self.n_noise_files)
                while set(noise_filenames) & set(files):
                    noise_filenames = random.sample(self.filenames, self.n_noise_files)

                x, edge_index, y, event_info, trigger_node = multi_load_graph(files[0], files[1:], noise_filenames, self.cylindrical_features_scale, self.phi_slope_max, self.z0_max, use_intt=self.use_intt, construct_edges=self.construct_edges, drop_l1=self.drop_l1, drop_l2=self.drop_l2, drop_l3=self.drop_l3, intt_filter=self.intt_filter, add_global_node=self.add_global_node)
            else:
                n_trigger = 1 + np.random.binomial(n_mix, 0.01)
                trigger_files = random.sample(self.filenames_1, n_trigger)
                nontrigger_files = random.sample(self.filenames_2, n_mix - n_trigger)
                files = trigger_files + nontrigger_files
                
                # Mix of both trigger and non-trigger
                noise_filenames = random.sample(self.filenames, self.n_noise_files)
                while set(noise_filenames) & set(files):
                    noise_filenames = random.sample(self.filenames, self.n_noise_files)


                x, edge_index, y,  event_info, trigger_node = multi_load_graph(files[0], files[1:], noise_filenames, self.cylindrical_features_scale, self.phi_slope_max, self.z0_max, use_intt=self.use_intt, construct_edges=self.construct_edges, drop_l1=self.drop_l1, drop_l2=self.drop_l2, drop_l3=self.drop_l3, intt_filter=self.intt_filter, add_global_node=self.add_global_node)




        w = y * self.real_weight + (1-y) * self.fake_weight
        start, end = edge_index
        w[(trigger_node[start] == 1) & (trigger_node[end] == 1)] *= self.trigger_edge_weight
        w[(trigger_node[start] == 2) & (trigger_node[end] == 2)] *= self.trigger_edge_weight

        if self.load_full_event:
            return torch_geometric.data.Data(
                    x=torch.from_numpy(x).to(torch.float),
                    edge_index=torch.from_numpy(edge_index).to(torch.long),
                    y=torch.from_numpy(y).to(torch.long), 
                    trigger=torch.from_numpy(event_info.trigger),
                    w=torch.from_numpy(w).to(torch.float),
                    trigger_node=trigger_node,
                    active_node=torch.from_numpy(event_info.active_node),
                    i=index, 
                    filename=event_file_name,
                    interaction_point=torch.from_numpy(event_info.interaction_point).to(torch.float).reshape(1, 3),
                    event_info=event_info
            )
        else:
            return torch_geometric.data.Data(
                    x=torch.from_numpy(x).to(torch.float),
                    edge_index=torch.from_numpy(edge_index).to(torch.long),
                    y=torch.from_numpy(y).to(torch.long), 
                    w=torch.from_numpy(w).to(torch.float),
                    trigger=torch.from_numpy(event_info.trigger),
                    trigger_node=trigger_node,
                    active_node=torch.from_numpy(event_info.active_node),
                    i=index, 
                    interaction_point=torch.from_numpy(event_info.interaction_point).to(torch.float).reshape(1, 3),
                    filename=event_file_name,
            )
       
    def __len__(self):
        return len(self.filenames)

    def preload(self):
        trigger_event_filenames = self.filenames_1 if self.f1_type else self.filenames_2
        non_trigger_event_filenames = self.filenames_2 if self.f1_type else self.filenames_1

        trigger_filename = np.random.choice(trigger_event_filenames)
        non_trigger_filename = np.random.choice(non_trigger_event_filenames)

        self.trigger_event_info = load_graph(trigger_filename, self.cylindrical_features_scale, self.phi_slope_max, self.z0_max, use_intt=self.use_intt, construct_edges=self.construct_edges, drop_l1=self.drop_l1, drop_l2=self.drop_l2, drop_l3=self.drop_l3, add_global_node=self.add_global_node)

        self.non_trigger_event_info = load_graph(non_trigger_filename, self.cylindrical_features_scale, self.phi_slope_max, self.z0_max, use_intt=self.use_intt, construct_edges=self.construct_edges, drop_l1=self.drop_l1, drop_l2=self.drop_l2, drop_l3=self.drop_l3, add_global_node=self.add_global_node)





    def get_positives(self, batch, use_preloaded=False):
        batch = batch.clone()
        active_nodes = batch.active_node.to(bool)
        inactive_nodes = ~active_nodes
        
        # Identify the batch indices and triggers for active nodes
        active_batch_indices = torch.unique(batch.batch[active_nodes], sorted=False)
        active_batch_triggers = batch.trigger[active_batch_indices]
        
        # Collect the node features and batch indices of the inactive nodes
        updated_node_features = batch.x[inactive_nodes]
        updated_batch_indices = batch.batch[inactive_nodes]
        num_inactive_nodes = updated_node_features.shape[0]
        
        new_node_features_list = []
        new_batch_indices_list = []
        
        # Determine filenames based on trigger type
        trigger_event_filenames = self.filenames_1 if self.f1_type else self.filenames_2
        non_trigger_event_filenames = self.filenames_2 if self.f1_type else self.filenames_1
        
        # Load new node features for each active batch index
        for batch_index_tensor, batch_trigger_value in zip(active_batch_indices, active_batch_triggers):
            batch_index = batch_index_tensor.item()
            if use_preloaded:
                if batch_trigger_value == 0:
                    new_node_features, edge_index, y, event_info = self.non_trigger_event_info
                else:
                    new_node_features, edge_index, y, event_info = self.trigger_event_info
            else:
                if batch_trigger_value == 0:
                    filename = np.random.choice(non_trigger_event_filenames)
                else:
                    filename = np.random.choice(trigger_event_filenames)
                
                new_node_features, edge_index, y, event_info = load_graph(
                    filename,
                    self.cylindrical_features_scale,
                    self.phi_slope_max,
                    self.z0_max,
                    use_intt=self.use_intt,
                    construct_edges=self.construct_edges,
                    drop_l1=self.drop_l1,
                    drop_l2=self.drop_l2,
                    drop_l3=self.drop_l3,
                    add_global_node=self.add_global_node
                )
            new_node_features_list.append(new_node_features)
            new_batch_indices_list.append(np.ones(new_node_features.shape[0], dtype=int) * batch_index)
            
        # Concatenate new node features and batch indices
        new_node_features_array = np.concatenate(new_node_features_list, axis=0)
        num_new_nodes = new_node_features_array.shape[0]
        
        updated_node_features = torch.cat(
            [updated_node_features, torch.tensor(new_node_features_array).to(updated_node_features.device)],
            dim=0
        ).to(updated_node_features.dtype)
        new_batch_indices_array = np.concatenate(new_batch_indices_list, axis=0)
        updated_batch_indices = torch.cat(
            [updated_batch_indices, torch.tensor(new_batch_indices_array).to(updated_batch_indices.device)],
            dim=0
        ).to(updated_batch_indices.dtype)
        
        # Update active_nodes: new nodes are active
        updated_active_nodes = torch.cat(
            [torch.zeros(num_inactive_nodes, dtype=torch.bool),
             torch.ones(num_new_nodes, dtype=torch.bool)],
            dim=0
        ).to(active_nodes.device)
        
        # Assign the updated tensors back to the batch object
        batch.x = updated_node_features
        batch.batch = updated_batch_indices
        batch.active_node = updated_active_nodes

        return batch

    def get_negatives(self, batch, use_preloaded=False):
        batch = batch.clone()
        active_nodes = batch.active_node.to(bool)
        inactive_nodes = ~active_nodes
        
        # Identify the batch indices and triggers for active nodes
        active_batch_indices = torch.unique(batch.batch[active_nodes], sorted=False)
        active_batch_triggers = batch.trigger[active_batch_indices]
        
        # Collect the node features and batch indices of the inactive nodes
        updated_node_features = batch.x[inactive_nodes]
        updated_batch_indices = batch.batch[inactive_nodes]
        num_inactive_nodes = updated_node_features.shape[0]
        
        new_node_features_list = []
        new_batch_indices_list = []
        
        # Determine filenames based on trigger type
        trigger_event_filenames = self.filenames_1 if self.f1_type else self.filenames_2
        non_trigger_event_filenames = self.filenames_2 if self.f1_type else self.filenames_1
        
        # Load new node features for each active batch index
        for batch_index_tensor, batch_trigger_value in zip(active_batch_indices, active_batch_triggers):
            batch_index = batch_index_tensor.item()
            if use_preloaded:
                if batch_trigger_value == 0:
                    new_node_features, edge_index, y, event_info = self.non_trigger_event_info
                else:
                    new_node_features, edge_index, y, event_info = self.trigger_event_info
            else:
                if batch_trigger_value == 0:
                    filename = np.random.choice(trigger_event_filenames)
                else:
                    filename = np.random.choice(non_trigger_event_filenames)
                
                new_node_features, edge_index, y, event_info = load_graph(
                    filename,
                    self.cylindrical_features_scale,
                    self.phi_slope_max,
                    self.z0_max,
                    use_intt=self.use_intt,
                    construct_edges=self.construct_edges,
                    drop_l1=self.drop_l1,
                    drop_l2=self.drop_l2,
                    drop_l3=self.drop_l3,
                    add_global_node=self.add_global_node
                )
            new_node_features_list.append(new_node_features)
            new_batch_indices_list.append(np.ones(new_node_features.shape[0], dtype=int) * batch_index)
            
        # Concatenate new node features and batch indices
        new_node_features_array = np.concatenate(new_node_features_list, axis=0)
        num_new_nodes = new_node_features_array.shape[0]
        
        updated_node_features = torch.cat(
            [updated_node_features, torch.tensor(new_node_features_array).to(updated_node_features.device)],
            dim=0 
        ).to(updated_node_features.dtype)
        new_batch_indices_array = np.concatenate(new_batch_indices_list, axis=0)
        updated_batch_indices = torch.cat(
            [updated_batch_indices, torch.tensor(new_batch_indices_array).to(updated_batch_indices.device)],
            dim=0
        ).to(updated_batch_indices.dtype)
        
        # Update active_nodes: new nodes are active
        updated_active_nodes = torch.cat(
            [torch.zeros(num_inactive_nodes, dtype=torch.bool),
             torch.ones(num_new_nodes, dtype=torch.bool)],
            dim=0
        ).to(active_nodes.device)
        
        # Assign the updated tensors back to the batch object
        batch.x = updated_node_features
        batch.batch = updated_batch_indices
        batch.active_node = updated_active_nodes
        batch.trigger = ~batch.trigger
        return batch



        


def get_datasets(n_train, n_valid, input_dir=None, filelist=None, real_weight=1.0, n_folders=1, input_dir2=None, phi_slope_max=0.03, z0_max=200, n_mix=1, use_intt=True, load_full_event=False, load_all=False, construct_edges=True, drop_l1=False, drop_l2=False, drop_l3=False, intt_filter=False, add_global_node=False, ramp_up_nmix=False, ramp_rate=1, random_n_mix=False, min_random_n_mix=1, trigger_edge_weight=1):
    data = HitGraphDataset(input_dir=input_dir, filelist=filelist,
        n_samples=n_train+n_valid, real_weight=real_weight, n_folders=n_folders, input_dir2=input_dir2, phi_slope_max=phi_slope_max, z0_max=z0_max, n_mix=n_mix, use_intt=use_intt, load_full_event=load_full_event, load_all=load_all, construct_edges=construct_edges, drop_l1=drop_l1, drop_l2=drop_l2, drop_l3=drop_l3, intt_filter=intt_filter, add_global_node=add_global_node, ramp_up_nmix=ramp_up_nmix, ramp_rate=ramp_rate, random_n_mix=random_n_mix, min_random_n_mix=min_random_n_mix, trigger_edge_weight=trigger_edge_weight)

    # Split into train and validation
    if load_all:
        n_train = len(data) 
        n_valid = 0

    if n_folders == 1:
        train_data, valid_data = random_split(data, [n_train, n_valid])
    if n_folders == 2:
        if load_all:
            train_data, valid_data = random_split(data, [n_train, n_valid])
        else:
            train_data, valid_data = random_split(data, [2*n_train, 2*n_valid])
    return train_data, valid_data
