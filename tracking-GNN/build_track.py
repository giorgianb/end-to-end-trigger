import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import os.path
import sys
import logging
import tqdm
import pickle
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from models.agnn_inference import GNNSegmentClassifier
from icecream import ic
from numpy.linalg import inv


def load_graph(file):
    with np.load(file, allow_pickle=True) as f:
        # print(list(f.keys()))
        hits = f['hits']
        scaled_hits = f['scaled_hits']
        hits_xyz = f['hits_xyz']
        noise_label = f['noise_label']
        layer_id = f['layer_id']
        edge_index = f['edge_index']
        pid = f['pid']
        n_hits = f['n_hits']
        n_tracks =f ['n_tracks']
        trigger_flag = f['trigger']
        ip = f['ip']
        psv = f['psv'] # secondary vertex
        p_momentum = f['p_momentum']
        e = f['e']
        valid_trigger_flag = f['valid_trigger_flag']
        ParticleTypeID = f['ParticleTypeID']
        is_complete_trk = f['is_complete_trk']
        trigger_track_flag = f['trigger_track_flag']
    return hits, scaled_hits, hits_xyz, noise_label, layer_id, edge_index, pid, n_hits, n_tracks, trigger_flag, ip, psv, p_momentum , e, valid_trigger_flag, ParticleTypeID, is_complete_trk, trigger_track_flag

# Used simply to find connected components
def dfs(node, label, seen, predicted_edges, layer_id):
    res = {}
    if node not in seen:
        # print('Add ' + str(node))
        
        # print('Seen ' + str(seen))
        seen.append(node)
        res[node] = label
        children = predicted_edges[predicted_edges[:, 0] == node][:, 1]
        for child in children:
            res = {**res, **dfs(child, label, seen, predicted_edges, layer_id)}

    # print('Return: ' + str(res))
    return res

def find_longest_root_to_leaf_path(nodes, predicted_edges, layer_id):
    def traverse_all_paths(node, path, paths):
        if node not in path:
            path.append(node)
            children = predicted_edges[predicted_edges[:, 0] == node][:, 1]
            children = children[layer_id[children] > layer_id[node]]
            if len(children) == 0:
                paths.append(path[:])
            else:
                for child in children:
                    traverse_all_paths(child, path, paths)
            path.pop()
                
        else:
            paths.append(path[:])

        
    longest_path = []
    for root in nodes:
        paths = []
        traverse_all_paths(root, [], paths)
        for path in paths:
            if len(path) > len(longest_path):
                longest_path = path
                
    return longest_path

def findMode(lst):
    return max(set(lst), key=lst.count)

def findModeList(lst):
    tuple_list = []
    for item in lst:
        if not item is None:
            tuple_list.append(tuple(item))

    return max(set(tuple_list), key=tuple_list.count)


def calculate_vector_length(v):
    return np.sqrt(sum(v**2))

def get_length(start, end):
    return np.sqrt(np.sum((start - end)**2, axis=1))

def calculate_edge_angle(v1, v2, v3):
    e1 = v2 - v1
    e2 = v3 - v2
    # print(e1)
    # print(e2)
    # print(calculate_vector_length(e1))
    # print(calculate_vector_length(e2))

    # if calculate_vector_length(e1) == 0 or calculate_vector_length(e2) == 0:c
    #     return
    return np.arccos(sum(e1 * e2)/calculate_vector_length(e1)/calculate_vector_length(e2))

def matmul_3D(A, B):
    return np.einsum('lij,ljk->lik', A, B)

def get_approximate_radii(tracks_info, n_hits, good_hits):
    x_indices = [3*j for j in range(5)]
    y_indices = [3*j+1 for j in range(5)]
    r = np.zeros((tracks_info.shape[0], 1))
    for n_hit in range(3, 5 + 1):
        complete_tracks = tracks_info[n_hits == n_hit]
        hit_indices = good_hits[n_hits == n_hit]
        if complete_tracks.shape[0] == 0:
            continue

        A = np.ones((complete_tracks.shape[0], n_hit, 3))
        x_values = complete_tracks[:, x_indices]
        x_values = x_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)

        y_values = complete_tracks[:, y_indices]
        y_values = y_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)
        A[:, :, 0] = x_values
        A[:, :, 1] = y_values

        y = - x_values**2 - y_values**2
        y = y.reshape((y.shape[0], y.shape[1], 1))
        AT = np.transpose(A, axes=(0, 2, 1))
        c = matmul_3D(matmul_3D(inv(matmul_3D(AT, A)), AT), y)
        r[n_hits == n_hit] == 1
        r[n_hits == n_hit] = np.sqrt(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])/200
    #test = get_approximate_radius(tracks_info, n_hits == 5)
    #assert np.allclose(test, r[n_hits == 5])

    return r

def filter_with_threshold(c, hits_xyz, layer_id):
    old_c = np.array(c)
    # Keep track of which "real" layers were present in the old component
    old_layers = set(layer_id[c]//2)

    c = np.array(c)
    track_hits = hits_xyz[c]
    X = np.expand_dims(track_hits[:, 0], -1)
    y = track_hits[:, 1:3]
    reg = LinearRegression().fit(X, y)
    y_p = reg.predict(X)
    diffs = np.sum((y_p - y)**2, axis=-1)

    keep = diffs <= THRESHOLDS[layer_id[old_c]]
    c = c[keep]
    new_layers = set(layer_id[c]//2)
    c = list(c)

    # If we have 3 layers in the old component, we want to make sure we at least have layers 1 and 2
    # in the new_component. That way we keep the possibility that the third hit might be a fluke
    # If we have 2 layers in the old component, we want to make sure we have at least layer 1 in the new component
    # That way we keep the possibility that the second hit might be a fluke
    if len(old_layers) > 1:
        to_check = (old_layers - new_layers) - {len(old_layers)}
    else:
        to_check = old_layers - new_layers

    for layer in to_check:
        candidates = old_c[layer_id[old_c]//2 == layer]
        candidate_index = np.argmin(diffs[layer_id[old_c]//2 == layer])
        c.append(candidates[candidate_index])
        
    return c

def get_connected_components(nodes, edges):
    connected_components = []
    cc = []
    def dfs(node):
        nonlocal cc, edges

        cc.append(node)
        children = edges[edges[:, 0] == node][:, 1]
        for child in children:
            if child not in cc:
                dfs(child)

        children = edges[edges[:, 1] == node][:, 0]
        for child in children:
            if child not in cc:
                dfs(child)


    visited = set()
    for node in nodes:
        if node in visited:
            continue
        cc = []
        dfs(node)
        assert visited & set(cc) == set()
        visited.update(cc)
        if len(cc) > 1:
            connected_components.append(cc)

    return {min(cc):list(set(cc)) for cc in connected_components}

sigmoid = torch.nn.Sigmoid()

def get_tracks(in_filename, data, dir_output, use_gt=False, save=False, with_threshold=False, with_longest_path=False, add_geo_features=True):
    hits, scaled_hits, hits_xyz, noise_label, layer_id, edge_index, pid, n_hits, n_tracks, trigger_flag, ip, psv, p_momentum, e, valid_trigger_flag, ParticleTypeID, is_complete_trk, trigger_track_flag = data
    # e = sigmoid(model([torch.tensor(scaled_hits[:, :3], dtype=torch.float), torch.tensor(edge_index, dtype=torch.long)])).detach().cpu().numpy().reshape(-1, 1)
    edge_candidates = np.transpose(edge_index)
    predicted_edges = edge_candidates[(e > 0.5).reshape(-1)]
    component_label = {i:i for i in range(hits.shape[0])}
    seen = []
    if not use_gt:
        res_continent = get_connected_components(range(hits.shape[0]), predicted_edges)
        # for node in range(hits.shape[0]):
        #     component_label = {**component_label, **dfs(node, node, seen, predicted_edges, layer_id)}

        # res_continent = {}
        # for node in component_label:
        #     label = component_label[node]
        #     if label in res_continent:
        #         res_continent[label].append(node)
        #     else:
        #             res_continent[label] = [node] 

        # labels = list(res_continent.keys())
        # for label in labels:
        #     if len(res_continent[label]) == 1:
        #             del res_continent[label]
        #     else:
        #         res_continent[label].sort()

         # Get momentum dict
        momentum_dict = {}
        for i in range(hits.shape[0]):
            if pid[i] not in momentum_dict:
                momentum_dict[pid[i]] = p_momentum[i]
        momentum_dict[-1] = np.array([0, 0, 0])

        ParticleTypeID_dict = {}
        for i in range(hits.shape[0]):
            if pid[i] not in ParticleTypeID_dict:
                ParticleTypeID_dict[pid[i]] = ParticleTypeID[i]
        ParticleTypeID[-1] = -1

        trigger_track_flag_dict = {}
        for i in range(hits.shape[0]):
            if pid[i] not in trigger_track_flag_dict:
                trigger_track_flag_dict[pid[i]] = trigger_track_flag[i]
        trigger_track_flag[-1] = False

    else:
        pid_dict = {}
        sv_dict = {}
        momentum_dict = {}
        ParticleTypeID_dict = {}
        trigger_track_flag_dict ={}
        for i in range(hits.shape[0]):
            if pid[i] in pid_dict:
                pid_dict[pid[i]].append(i)
            else:
                pid_dict[pid[i]] = [i]
                sv_dict[pid[i]] = psv[i]
                momentum_dict[pid[i]] = p_momentum[i]
                ParticleTypeID_dict[pid[i]] = ParticleTypeID[i]
                trigger_track_flag_dict[pid[i]] = trigger_track_flag[i]

        res_continent = pid_dict
        if -1 in res_continent:
            del res_continent[-1]
    
    # Generate Track Vector
    i = 0
    count_incomplete_tracks = 0
    k = len(res_continent)
    track_vector = np.zeros((k, 15))
    hits_coord = hits_xyz[:, :3]
    hits_center = np.mean(hits_coord, axis=0)

    list_psv = []
    list_pid = []
    is_complete = []

    # print('n_tracks: ' + str(len(res_continent)))
    # organize the track info from the edge prediction
    # 5 hits + 4 edge length + 1 total length, 1 angle + 4 delta angle, hits center , total 15+13

    # ic(res_continent)

    for cid in res_continent:
        # print(cid)
        c = res_continent[cid]
        if with_longest_path:
            c = find_longest_root_to_leaf_path(c, predicted_edges, layer_id)
            assert len(res_continent[cid]) == 0 or len(c) != 0, "Zero-length root-to-leaf path when this should not be the case"
            if use_gt and set(res_continent[cid]) != set(c):
                print(f'Warning: could not construct true track using ground truth for event {in_filename}')

        if with_threshold:
            # Keep track of what the old component was
            c = filter_with_threshold(c, hits_xyz, layer_id)
                 
        count_incomplete_flag = True
        
        # track coord according to layer_id
        if (layer_id[c] == 0).any():
            mask = (layer_id[c] == 0).reshape(-1)
            track_vector[i, :3] = np.mean(hits_coord[c][mask], axis = 0)
        else:
            count_incomplete_flag = False
        
        if (layer_id[c] == 1).any():
            mask = (layer_id[c] == 1).reshape(-1)
            track_vector[i, 3:6] = np.mean(hits_coord[c][mask], axis = 0)
        else:
            count_incomplete_flag = False
        
        if (layer_id[c] == 2).any():
            mask = (layer_id[c] == 2).reshape(-1)
            track_vector[i, 6:9] = np.mean(hits_coord[c][mask], axis = 0)
        else:
            count_incomplete_flag = False
        
        if np.logical_or(layer_id[c] == 3, layer_id[c] == 4).any():
            mask = (np.logical_or(layer_id[c] == 3, layer_id[c] == 4)).reshape(-1)
            track_vector[i, 9:12] = np.mean(hits_coord[c][mask], axis = 0)
        else:
            count_incomplete_flag = False
        
        if np.logical_or(layer_id[c] == 5, layer_id[c] == 6).any():
            mask = (np.logical_or(layer_id[c] == 5, layer_id[c] == 6)).reshape(-1)
            track_vector[i, 12:15] = np.mean(hits_coord[c][mask], axis = 0)
        else:
            count_incomplete_flag = False
        
        # geometric features
#         track_vector[i, 15] = max(calculate_vector_length(track_vector[i, 3:6]-track_vector[i, 0:3]), 0)
#         track_vector[i, 16] = max(calculate_vector_length(track_vector[i, 6:9]-track_vector[i, 3:6]), 0)

        if count_incomplete_flag:
            count_incomplete_tracks += 1
        is_complete.append(count_incomplete_flag)
#         else:
#             track_vector[i, 17] = calculate_edge_angle(track_vector[i, 0:3], track_vector[i,3:6],                 track_vector[i, 6:9])
#         track_vector[i, 12] = calculate_vector_length(hits_coord[c[0]]-hits_coord[c[-1]])


        # To mode psv
        if any(psv[c]):
            mode_psv = findModeList(psv[c])
        else:
            mode_psv = ip
        list_psv.append(np.array(mode_psv))
        
        # To mode pid
        mode_pid = findMode(list(pid[c]))
        list_pid.append(mode_pid)

        i += 1
        
#     track_vector[:, 13:16] = hits_center
    if track_vector.shape[0] != 0 and add_geo_features:
        # 4 edge length + 1 total length, 1 angle + 4 delta angle, hits center , total 13
        geo_features = np.zeros((track_vector.shape[0], 13))
        phi  = np.zeros((track_vector.shape[0], 5))
        geo_features[:, 5] = np.arctan2(track_vector[:, 1], track_vector[:, 0])
        for i in range(4):
            geo_features[:, i] = get_length(track_vector[:, (3*i+3):(3*i+6)], track_vector[:, (3*i):(3*i+3)])
        for i in range(5):
            phi[:, i] = np.arctan2(track_vector[:, (3*i)+1], track_vector[:, (3*i)])
        geo_features[:, 5] = get_length(track_vector[:, 12:15], track_vector[:, 0:3])
        geo_features[:, 6:10] = np.diff(phi)
        geo_features[:, 10:13] = hits_center
        track_vector = np.concatenate([track_vector, geo_features], axis=1)
    
    # ic(track_vector)

    dict_list_psv = {}
    for itm in list_psv:
        if not tuple(itm) in dict_list_psv:
            dict_list_psv[tuple(itm)] = len(dict_list_psv)

    res = {}
    res['n_tracks'] = np.array(track_vector.shape[0])
    res['tracks_info'] = track_vector
    res['original_track_label'] = np.array([dict_list_psv[tuple(itm)] for itm in list_psv])
    res['pid'] = np.array(list_pid)
    res['momentum'] = np.array([momentum_dict[k] for k in list_pid])
    res['is_trigger_track'] = np.array([trigger_track_flag_dict[k] for k in list_pid])
    res['ParticleTypeID'] = np.array([ParticleTypeID_dict[k] for k in list_pid])
    res['trigger_flag'] = np.array(trigger_flag)
    res['track_2nd_vertex'] = np.array(list_psv)
    res['modified_2nd_vertex'] = res['track_2nd_vertex']
    res['ip'] = ip
    res['valid_trigger_flag'] = valid_trigger_flag
        
#     res['tracks_to_hits'] = res_continent
    
    # is_trigger_track = []
    # if track_vector.shape[0] != 0:
    #     is_trigger_track = (np.isclose(res['track_2nd_vertex'], ip, atol=0.001).all(axis=1)) == False
    # res['is_trigger_track'] = is_trigger_track
    
    res['is_complete'] = is_complete
    
    # calculate radius
    hits = track_vector[:, :15].reshape(track_vector.shape[0], 5, 3)
    good_hits = np.all(hits != 0, axis=-1)
    n_hits = np.sum(good_hits, axis=-1)
    r = get_approximate_radii(track_vector, n_hits, good_hits)
    res['r'] = r
    
    res['n_hits'] = n_hits
    # print(res)
    if save:
        out_filename = os.path.join(dir_output, in_filename.split('/')[-1])
        np.savez(out_filename, **res) 
    
    return res
            
        

input_dir = '/ssd2/tingting/alltrack_predicted_edge/trigger/1'
output_dir = '/ssd2/tingting/alltrack_predicted_trk/trigger/1'
start_file = 500000
n_files = 500000
os.makedirs(output_dir, exist_ok=True)
    
filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith('event')])[start_file:start_file+n_files]
print(f'Input Dir: {input_dir}')

for i, filename in enumerate(filenames):
    # ic(filename)
    data = load_graph(filename)
    res = get_tracks(filename, data, output_dir, save=True, use_gt=False, with_longest_path=False, with_threshold=False, add_geo_features=False)
    if i % 1000 == 0:
        ic(i, filename)
print('done!')

