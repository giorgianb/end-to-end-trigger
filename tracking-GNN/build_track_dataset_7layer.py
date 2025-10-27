#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np
import os
import os.path
import sys
import tqdm
import pickle
from collections import defaultdict
from icecream import ic
from numpy.linalg import inv
import csv
from itertools import chain, combinations

from joblib import Parallel, delayed
import contextlib
import joblib
from tqdm import tqdm

# Thresholds for deviation from line
THRESHOLDS = np.array([0.007230312233600477, 0.03609715402879862, 0.03732014063230461, 0.11144256888225307, 0.010710058987408799, 0.028489891702947225])
#THRESHOLDS = np.zeros(6)

def load_graph(file):
    with np.load(file, allow_pickle=True) as f:
        # print(list(f.keys()))
        hits = f['hits'] if 'hits' in f else f['x']
        scaled_hits = f['scaled_hits']
        hits_xyz = f['hits_xyz']
        noise_label = f['noise_label']
        layer_id = f['layer_id']
        edge_index = f['edge_index']
        pid = f['pid']
        #pid_1 = f['pid_1']
        n_hits = f['n_hits']
        n_tracks =f ['n_tracks']
        trigger_flag = f['trigger']
        ip = f['ip']
        psv = f['psv'] # secondary vertex
        p_momentum = f['p_momentum']
        valid_trigger_flag = f['valid_trigger_flag']
        ParticleTypeID = f['ParticleTypeID']
        is_complete_trk = f['is_complete_trk']
        trigger_track_flag = f['trigger_track_flag']
        # In case we are working directly on the edge candidates
        if 'e' in f.keys():
            e = f['e']
        else:
            e = None
    return hits, scaled_hits, hits_xyz, noise_label, layer_id, edge_index, pid, n_hits, n_tracks, trigger_flag, ip, psv, p_momentum, e, valid_trigger_flag, ParticleTypeID, is_complete_trk, trigger_track_flag

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


# Power-set in reverse order
def powerset(iterable):
    s = list(iterable)
    length = min(7, len(s))
    return chain.from_iterable(combinations(s, r) for r in reversed(range(2, length+1)))

def is_connected(nodes, predicted_edges):
    visited = set()
    def dfs(node):
        nonlocal visited
        if node in visited:
            return
        visited.add(node)
        children = predicted_edges[predicted_edges[:, 0] == node][:, 1]
        for child in children:
            if child not in visited:
                dfs(child)


        children = predicted_edges[predicted_edges[:, 1] == node][:, 0]
        for child in children:
            if child not in visited:
                dfs(child)


    dfs(nodes[0])
    return all(node in visited for node in nodes)


def find_all_paths(nodes, predicted_edges, layer_id, min_length=1, max_paths=32):
    # TODO: remove. We just need to quickly generate connected components
    # TODO: remove. For now we are just trying to match pixel tracks
    return [list(nodes)]
    #if len(nodes) > 1:
    #    return [list(nodes)]
    #else:
    #    return []

    paths = []
    for subset in powerset(sorted(nodes)):
        if len(subset) < min_length:
            return paths
        elif len(paths) == max_paths:
            return paths
        #elif not is_connected(subset, predicted_edges):
        #    continue

        paths.append(list(subset))

    return paths
        

def find_all_paths_old(nodes, predicted_edges, layer_id):
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

        
    all_paths = []
    root_layer_id = None
    # TODO: also experiment with keeping all the paths instead of just the ones that start
    # at the lowest layer
    for root in nodes:
        paths = []
        traverse_all_paths(root, [], paths)
        if root_layer_id is None:
            all_paths.extend(paths)
            root_layer_id = layer_id[root]
        elif layer_id[root] < root_layer_id:
            all_paths = paths
            root_layer_id = layer_id[root]
                
    return all_paths



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

def get_line(hit_ids, layer_id, hits):
    A = np.ones((hits.shape[0], 2), dtype=np.float64)
    A[:, 0] = hits[:, 0]
    b = hits[:, 1]
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    errors = b - A @ x
    diffs = np.zeros(5)
    for i in range(diffs.shape[0]):
        diffs[i] = np.sum(errors[layer_id[hit_ids] == i])

    return x, diffs

def get_radius(hits):
    A = np.ones((1, len(hits), 3))
    A[0, :, 0] = hits[:, 0]
    A[0, :, 1] = hits[:, 1]
    y = -(hits[:, 0]**2 + hits[:, 1]**2)
    y = y.reshape((1, y.shape[0], 1))
    AT = np.transpose(A, axes=(0, 2, 1))
    # print(A.shape, AT.shape, y.shape)
    # c = inv(matmul_3D(A, AT))
    c = matmul_3D(matmul_3D(inv(matmul_3D(AT, A)), AT), y)
    # print(A.shape, AT.shape, y.shape, c.shape)
    if c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2] < 0:
        ic(hits)
        ic(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])
    r = np.sqrt(np.abs(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2]))/200
    return r[0]

def get_approximate_radius_and_fit(hit_ids, layer_id, hits):
    A = np.ones((1, len(hits), 3))
    A[0, :, 0] = hits[:, 0]
    A[0, :, 1] = hits[:, 1]
    y = -(hits[:, 0]**2 + hits[:, 1]**2)
    y = y.reshape((1, y.shape[0], 1))
    AT = np.transpose(A, axes=(0, 2, 1))
    # print(A.shape, AT.shape, y.shape)
    # c = inv(matmul_3D(A, AT))
    c = matmul_3D(matmul_3D(inv(matmul_3D(AT, A)), AT), y)
    # print(A.shape, AT.shape, y.shape, c.shape)

    # TODO: The abs is a hack to get rid of sometimes-negative values
    r = np.sqrt(np.abs(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2]))/200
    center = -c[0, :2, 0]/2
    distances = np.linalg.norm(hits[:, :2] - center, axis=-1)
    errors = np.abs(distances - r[0])
    diffs = np.zeros(5)
    for i in range(diffs.shape[0]):
        diffs[i] = np.sum(errors[layer_id[hit_ids] == i])

    return r[0], diffs

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
    from sklearn.linear_model import LinearRegression
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

def get_res_cid(data, use_gt=True):
    hits, scaled_hits, hits_xyz, noise_label, layer_id, edge_index, pid_1, n_hits, n_tracks, trigger_flag, ip, psv, p_momentum, e, valid_trigger_flag, ParticleTypeID, is_complete_trk, trigger_track_flag = data
    edge_candidates = np.transpose(edge_index)
    predicted_edges = edge_candidates[(e > 0.5).reshape(-1)]
    component_label = {i:i for i in range(hits.shape[0])}
    seen = []
    if not use_gt:
        for node in range(hits.shape[0]):
            component_label = {**component_label, **dfs(node, node, seen, predicted_edges, layer_id)}

            res_continent = {}
            for node in component_label:
                label = component_label[node]
                if label in res_continent:
                    res_continent[label].append(node)
                else:
                     res_continent[label] = [node] 

            labels = list(res_continent.keys())
            for label in labels:
                if len(res_continent[label]) == 1:
                     del res_continent[label]
                else:
                    res_continent[label].sort()
         # Get momentum dict
        momentum_dict = {}
        for i in range(hits.shape[0]):
            if pid[i] not in momentum_dict:
                momentum_dict[pid[i]] = p_momentum[i]
        momentum_dict[-1] = np.array([0, 0, 0])
    else:
        pid_dict = {}
        sv_dict = {}
        momentum_dict = {}
        for i in range(hits.shape[0]):
            if pid[i] in pid_dict:
                pid_dict[pid[i]].append(i)
            else:
                pid_dict[pid[i]] = [i]
                sv_dict[pid[i]] = psv[i]
                momentum_dict[pid[i]] = p_momentum[i]

            if pid_1[i] >= 0 and pid_1[i] in pid_dict:
                pid_dict[pid_1[i]].append(i)
            elif pid_1[i] >= 0:
                pid_dict[pid[i]] = [i]
                sv_dict[pid[i]] = psv[i]
                momentum_dict[pid[i]] = p_momentum[i]


                
        res_continent = pid_dict
        if -1 in res_continent:
            del res_continent[-1]

    data = []
    for cid, hits in res_continent.items():
        momentum = np.mean(np.stack(p_momentum[hits], axis=0), axis=0)
        length= len(hits)
        r = get_radius(hits_xyz[hits])
        data.append((length, momentum, r))

    return res_continent, data

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

    return {min(cc):cc for cc in connected_components}


 

def get_tracks(in_filename, data, dir_output, use_gt=False, save=False, add_geo_features=True):
    hits, scaled_hits, hits_xyz, noise_label, layer_id, edge_index, pid_1, n_hits, n_tracks, trigger_flag, ip, psv, p_momentum, e = data
    if len(layer_id.shape) == 2:
        layer_id = layer_id.squeeze(-1)

    edge_candidates = np.transpose(edge_index)
        predicted_edges = edge_candidates[(e > 0.5).reshape(-1)]
    component_label = {i:i for i in range(hits.shape[0])}
    seen = []
    #if not use_gt:
    #    res_continent = get_connected_components(range(hits.shape[0]), predicted_edges)
    #     # Get momentum dict
    #    momentum_dict = {}
    #    for i in range(hits.shape[0]):
    #        if pid[i] not in momentum_dict:
    #            momentum_dict[pid[i]] = p_momentum[i]
    #    momentum_dict[-1] = np.array([0, 0, 0])
    #else:
    if True:
        pid_dict = {}
        sv_dict = {}
        momentum_dict = {}
        for i in range(hits.shape[0]):
            #print(f'Processing hit {i}: pid: {pid[i]} pid1: {pid_1[i]} null p_momentum: {p_momentum[i] is None} low p_t: {p_momentum[i][0]**2 + p_momentum[i][1]**2 <= 0.04}')
            if p_momentum[i] is not None and p_momentum[i][0]**2 + p_momentum[i][1]**2 >= 0.04 and pid[i] > 0:
                if pid[i] in pid_dict:
                    pid_dict[pid[i]].append(i)
                    #print(f'Adding hit {i} to track {pid[i]}')
                    if pid[i] not in sv_dict:
                        sv_dict[pid[i]] = psv[i]
                        momentum_dict[pid[i]] = p_momentum[i]
                else:
                    #print(f'Adding hit {i} to track {pid[i]}')
                    pid_dict[pid[i]] = [i]
                    sv_dict[pid[i]] = psv[i]
                    momentum_dict[pid[i]] = p_momentum[i]

            if pid_1[i] >= 0 and pid_1[i] in pid_dict:
                #print(f'Adding hit {i} to track {pid_1[i]}')
                pid_dict[pid_1[i]].append(i)
            elif pid_1[i] >= 0:
                #print(f'Adding hit {i} to track {pid_1[i]}')
                pid_dict[pid_1[i]] = [i]

        to_delete = tuple(filter(lambda x: x not in sv_dict.keys(), pid_dict.keys()))
        for item in to_delete:
            del pid_dict[item]

        for item in sv_dict:
            assert item in momentum_dict
                
        res_continent = pid_dict
        if -1 in res_continent:
            del res_continent[-1]
        #print(res_continent)
    
    # Generate Track Vector
    i = 0
    count_incomplete_tracks = 0
    k = len(res_continent)
    hits_coord = hits_xyz[:, :3]
    hits_pixels = np.expand_dims(hits[:, -1], -1)
    hits_center = np.mean(hits_coord, axis=0)


    # print('n_tracks: ' + str(len(res_continent)))
    # organize the track info from the edge prediction
    # 5 hits + 4 edge length + 1 total length, 1 angle + 4 delta angle, hits center , total 15+13
    all_track_vectors = []
    all_list_psv = []
    all_is_complete = []
    all_list_pid = []
    all_radii = []
    all_cids = []
    all_original_track_labels = []
    n_paths = []
    all_momentums = []
    lengths = []
    for cid in res_continent:
        # print(cid)
        track_vectors = []
        list_psv = []
        list_pid = []
        is_complete = []
        rs = []
        diffs = []
        xs = []
        diff_xs = []
        ys = []
        diff_ys = []

        c = res_continent[cid]
        component_hits = hits_coord[c][:, :2]
        #print(component_hits.shape)
        thetas = np.arctan2(component_hits[:, 1], component_hits[:, 0])
        # Ignore weird tracks

        # TODO: add back
        #if theta_range(thetas) >= 2*np.pi/10:
        #    continue

        # TODO: uncomment. For now we are just trying to match tracks
        #if len(res_continent[cid]) <= 1:
        #    continue

        if use_gt:
            paths = [c]
        else:
            paths = find_all_paths(c, predicted_edges, layer_id, min_length=2)
            assert len(res_continent[cid]) <= 1 or len(paths) != 0, f"No root-to-leaf paths found when this should not be the case ({res_continent[cid]}), ({paths})"

        n_paths.append(len(paths))
        for c in paths:
            local_layer_id = np.array(layer_id[c])
            local_layer_id[local_layer_id == 4] = 3
            local_layer_id[local_layer_id == 6] = 5
            if len(set(local_layer_id)) >= 3:
                #r, diff = get_approximate_radius_and_fit(c, layer_id, hits_xyz[c])
                r = get_radius(hits_xyz[c])
                diff = np.zeros(5)
            else:
                r, diff = np.array([0.0]), np.zeros(5)

            if len(c) >= 2:
                x, diff_x = get_line(c, layer_id, hits_xyz[c][:, [0, 2]])
                y, diff_y = get_line(c, layer_id, hits_xyz[c][:, [1, 2]])
            else:
                x, diff_x = np.array([0.0, 0.0]), np.zeros(5)
                y, diff_y = np.array([0.0, 0.0]), np.zeros(5)
            
            rs.append(r)
            diffs.append(diff)

            xs.append(x)
            diff_xs.append(diff_x)

            ys.append(y)
            diff_ys.append(diff_y)

            track_vector = np.zeros(15)
            count_incomplete_flag = True
            
            # track coord according to layer_id
            if (layer_id[c] == 0).any():
                mask = layer_id[c] == 0
                #track_vector[:3] = np.mean(hits_coord[c][mask], axis = 0)
                track_vector[:3] = np.sum((hits_pixels[c][mask] * hits_coord[c][mask])/np.sum(hits_pixels[c][mask]), axis=0)
            else:
                count_incomplete_flag = False
            
            if (layer_id[c] == 1).any():
                mask = layer_id[c] == 1
                #track_vector[3:6] = np.mean(hits_coord[c][mask], axis = 0)
                track_vector[3:6] = np.sum((hits_pixels[c][mask] * hits_coord[c][mask])/np.sum(hits_pixels[c][mask]), axis=0)
            else:
                count_incomplete_flag = False
            
            if (layer_id[c] == 2).any():
                mask = layer_id[c] == 2
                #track_vector[6:9] = np.mean(hits_coord[c][mask], axis = 0)
                track_vector[6:9] = np.sum((hits_pixels[c][mask] * hits_coord[c][mask])/np.sum(hits_pixels[c][mask]), axis=0)
            else:
                count_incomplete_flag = False
            
            if np.logical_or(layer_id[c] == 3, layer_id[c] == 4).any():
                mask = np.logical_or(layer_id[c] == 3, layer_id[c] == 4)
                #track_vector[9:12] = np.mean(hits_coord[c][mask], axis = 0)
                track_vector[9:12] = np.sum((hits_pixels[c][mask] * hits_coord[c][mask])/np.sum(hits_pixels[c][mask]), axis=0)
            else:
                count_incomplete_flag = False
            
            if np.logical_or(layer_id[c] == 5, layer_id[c] == 6).any():
                mask = np.logical_or(layer_id[c] == 5, layer_id[c] == 6)
                #track_vector[12:15] = np.mean(hits_coord[c][mask], axis = 0)
                track_vector[12:15] = np.sum((hits_pixels[c][mask] * hits_coord[c][mask])/np.sum(hits_pixels[c][mask]), axis=0)
            else:
                count_incomplete_flag = False

            if count_incomplete_flag:
                count_incomplete_tracks += 1
            is_complete.append(count_incomplete_flag)

            # To mode psv
            if any(psv[c]):
                mode_psv = findModeList(psv[c])
            else:
                mode_psv = ip
            list_psv.append(np.array(mode_psv))
            
            # To mode pid
            mode_pid = findMode(list(filter(lambda x: x >= 0, list(pid[c]) + list(pid_1[c]))))
            list_pid.append(mode_pid)
            track_vectors.append(track_vector)
            lengths.append(len(track_vectors))

        track_vectors = np.stack(track_vectors, axis=0)
        rs = np.stack(rs, axis=0)
        diffs = np.stack(diffs, axis=0)
        xs = np.stack(xs, axis=0)
        diff_xs = np.stack(diff_xs, axis=0)
        ys = np.stack(ys, axis=0)
        diff_ys = np.stack(diff_ys, axis=0)

        if track_vectors.shape[0] != 0 and add_geo_features:
            # 4 edge length + 1 total length, 1 angle + 4 delta angle, hits center , total 13
            geo_features = np.zeros((track_vectors.shape[0], 13))
            phi  = np.zeros((track_vectors.shape[0], 5))
            geo_features[:, 5] = np.arctan2(track_vectors[:, 1], track_vectors[:, 0])
            for i in range(4):
                geo_features[:, i] = get_length(track_vectors[:, (3*i+3):(3*i+6)], track_vectors[:, (3*i):(3*i+3)])
            for i in range(5):
                phi[:, i] = np.arctan2(track_vectors[:, (3*i)+1], track_vectors[:, (3*i)])
            geo_features[:, 5] = get_length(track_vectors[:, 12:15], track_vectors[:, 0:3])
            geo_features[:, 6:10] = np.diff(phi)
            geo_features[:, 10:13] = hits_center
            track_vectors = np.concatenate([track_vectors, geo_features], axis=1)

        #track_vectors = np.concatenate([rs, diffs, xs, diff_xs, ys, diff_ys, track_vectors], axis=-1)
        track_vectors = np.concatenate([rs, track_vectors], axis=-1)

        all_cids.extend([cid] * track_vectors.shape[0])
        all_list_pid.append(list_pid)
        all_list_psv.append(list_psv)
        all_track_vectors.append(track_vectors)
        all_is_complete.append(is_complete)
        all_radii.append(r)
    
        dict_list_psv = {}
        for itm in list_psv:
            if not tuple(itm) in dict_list_psv:
                dict_list_psv[tuple(itm)] = len(dict_list_psv)

        def nullify(x):
            return x if x is not None else [float('NaN'), float('NaN'), float('NaN')]

        all_original_track_labels.append(np.array([dict_list_psv[tuple(itm)] for itm in list_psv]))
        try:
            all_momentums.append(np.array([nullify(momentum_dict[k]) for k in list_pid]))
        except KeyError:
            ic(pid[c])
            ic(momentum_dict)
            ic(list_pid)
            raise

    res = {}

    if len(all_track_vectors) != 0:
        res['tracks_info'] = np.concatenate(all_track_vectors, axis=0)
        res['n_tracks'] = np.array(res['tracks_info'].shape[0])
        res['original_track_label'] = np.concatenate(all_original_track_labels, axis=0)
        res['pid'] = np.concatenate(all_list_pid, axis=0)
        res['momentum'] = np.concatenate(all_momentums, axis=0)
        res['trigger_flag'] = np.array(trigger_flag)
        res['track_2nd_vertex'] = np.concatenate(all_list_psv, axis=0)
        res['modified_2nd_vertex'] = res['track_2nd_vertex']
        res['ip'] = ip
        res['cids'] = np.array(all_cids)

        res['tracks_to_hits'] = res_continent
        
        is_trigger_track = []
        if track_vector.shape[0] != 0:
            is_trigger_track = (np.isclose(res['track_2nd_vertex'], ip, atol=0.001).all(axis=1)) == False
        res['is_trigger_track'] = is_trigger_track
        res['is_complete'] = np.stack(all_is_complete, axis=0)
        
        # calculate radius
        res['r'] = np.concatenate(all_radii, axis=0)
        
        res['n_hits'] = n_hits
    else:
        res['tracks_info'] = np.zeros((0, 29))
        res['n_tracks'] = 0
        res['original_track_label'] = np.zeros(0)
        res['pid'] = np.zeros(0)
        res['momentum'] = np.zeros((0, 3))
        res['trigger_flag'] = np.array(trigger_flag)
        res['track_2nd_vertex'] = np.zeros((0, 3))
        res['modified_2nd_vertex'] = res['track_2nd_vertex']
        res['ip'] = np.zeros((0, 3))
        res['cids'] = np.zeros(0)

        res['tracks_to_hits'] = res_continent
        
        res['is_trigger_track'] = np.zeros(0)
        res['is_complete'] = np.zeros((0, 1))
        
        # calculate radius
        res['r'] = np.zeros(0)
        
        res['n_hits'] = n_hits
        
    if save:
        gt_res, _ = get_tracks(in_filename, data, "", use_gt=True, save=False, add_geo_features=add_geo_features)
        for key, value in gt_res.items():
            res[f'gt_{key}'] = value

        out_filename = os.path.join(dir_output, in_filename.split('/')[-1])
        np.savez(out_filename, **res) 
    
    return res, lengths
            
        
        

is_trigger = True
if is_trigger:
    #input_dir = '/home/tingtingxuan/Data/7layer_data/tracking_inference_INTTclustered_postrack2/trigger'
    #input_dir = '/home/giorgian/data/parsed-4x/trigger'
    input_dir = '/ssd1/giorgian/alltrack_predicted_edge/trigger/1/'
    output_dir = '/ssd1/giorgian/data/alltrack/trigger/1/'
    #output_dir = '/home/giorgian/data/gt-4x/trigger'
else:
    #input_dir = '/home/giorgian/data/parsed-4x/non-trigger'
    #output_dir = '/home/giorgian/data/gt-4x/non-trigger'
    #input_dir = '/home1/giorgian/projects/parsing/parsed-nocluster/non-trigger'

    #input_dir = '/ssd2/giorgian/HFMLNewFiles-hits-1/nontrigger/0/'
    input_dir = '/ssd1/giorgian/alltrack_predicted_edge/trigger/1/'
    output_dir = '/ssd1/giorgian/data/alltrack/trigger/0/'
    #output_dir = '/ssd1/giorgian/data/gt-nocluster-1/non-trigger'

    #input_dir = '/home/tingtingxuan/Data/7layer_data/tracking_inference_INTTclustered_postrack2/nontrigger'

    #input_dir = '/home1/giorgian/projects/parsing/parsed-nocluster/non-trigger'
    #output_dir = '/home/giorgian/data/gt-4x/non-trigger'
    #output_dir = '/ssd1/giorgian/data/gt-nocluster/trigger'


def theta_range(thetas):
    thetas = np.array(thetas)
    thetas[thetas < 0] += 2*np.pi
    thetas = np.sort(thetas)
    #print('Thetas: ', thetas)
    min_range = np.inf
    for i in range(len(thetas) - 1):
        # Because these points are adjacent
        t1 = thetas[i]
        t2 = thetas[i + 1]
        min_range = min(min_range, 2*np.pi - (t2 - t1))

    min_range = min(min_range, thetas[-1] - thetas[0])
    return min_range

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


if __name__ == '__main__':
    filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith('event')])
    #filenames = ['/home/giorgian/data/parsed-4x/trigger/event100051200.npz']
#    completed = False
#    all_lengths = []
#    with open('trigger-weird.txt', 'w') as handle:
#        for i, filename in enumerate(tqdm.tqdm(filenames)):
#            data = load_graph(filename)
#            res, _ = get_res_cid(data, use_gt=True)
#            hits = data[2]
#            for pid, hit_ids in res.items():
#                h = hits[hit_ids]
#                thetas = np.arctan2(h[:, 1], h[:, 0]) 
#                if theta_range(thetas) >= 2*np.pi/10:
#                    print(filename, file=handle)

#    with open('non-trigger-stats.csv', 'w', newline='') as handle:
#        writer = csv.writer(handle)
#        for i, filename in enumerate(tqdm.tqdm(filenames)):
#            data = load_graph(filename)
#            _, results = get_res_cid(data)
#            for d in results:
#                writer.writerow([is_trigger, i, d[0]] + d[1].tolist() + [d[2][0]])
#
#        #tracks, lengths = get_tracks(filename, data, output_dir, False, True, True)
#        #all_lengths.extend(lengths)
#
#    #print(sum(all_lengths)/len(all_lengths))
    def process(filename):
        try:
            data = load_graph(filename)
            get_tracks(filename, data, output_dir, use_gt=False, save=True, add_geo_features=True)
        except Exception as e:
            raise
            print(e)

    print(f"[Trigger = {is_trigger}]")
    with tqdm_joblib(tqdm(desc="Conversion", total=len(filenames))) as progress_bar:
        Parallel(n_jobs=16)(delayed(process)(filename) for filename in filenames)
