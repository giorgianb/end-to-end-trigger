import os
import numpy as np

def load_graph(filename):

    with np.load(filename) as f:
        # trigger_flag = f['trigger']
        # edge_index = f['edge_index']
        # hits = f['hits']
        # # hits = np.concatenate([hits, np.reshape([hits.shape[0]] * hits.shape[0], (-1, 1))], 1)
        # e = f['e']
        # hits_fix = np.zeros([408, 4])
        # n_hit = hits.shape[0]
        # hits_fix[:n_hit, :] = hits

        # edge_index_fix = np.zeros([2, 1388])
        # n_edge = edge_index.shape[1]
        # edge_index_fix[:, :n_edge] = edge_index

        # e_fix = np.zeros([1388, 1])
        # e_fix[:n_edge, :] = e
        n_hit = f['hits'].shape[0]
        n_edge = f['edge_index'].shape[1]
    return n_edge, n_hit

n = 130000
input_dir = '/home/tingting/Data/tracking_result_allinfo_7layer/tracking_result_with_noise_In_dim32'
filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                    if f.startswith('event') and not f.endswith('_ID.npz')])[:n]
input_dir = '/home/tingting/Data/tracking_result_allinfo_7layer/tracking_result_with_noise_D0_dim32'
filenames += sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                    if f.startswith('event') and not f.endswith('_ID.npz')])[:n]

print(len(filenames), ' files will be loaded.')

hits_info = []
edge_indexs = []
trig = []
es = []
n_hits = []
n_edges = []

for file_index in range(len(filenames)):
    n_edge, n_hit = load_graph(filenames[file_index])
    # hits_info.append(hits)
    # edge_indexs.append(edge_index)
    # trig.append(trigger_flag)
    # es.append(e)
    n_hits.append(n_hit)
    n_edges.append(n_edge)
    if file_index%1000 == 0:
        print(file_index, ' files has been loaded.')

print(max(n_hits), max(n_edges))

# hits_info = np.array(hits_info)
# edge_index = np.array(edge_index)
# trig = np.array(trig)
# e = np.array(e)
# n_hits = np.array(n_hits)
# n_edges = np.array(n_edges)

# print(hits_info.shape)
# print(hits_info[0].shape)
# print(hits_info[0][:n_hits[0]].shape)

# np.savez('/home/tingting/kdd2022/trigger_pred/data/mixed_data_7layer_260k', hits_info=hits_info, edge_index=edge_indexs, trig=trig, e=es, n_hits=n_hits, n_edges=n_edges)