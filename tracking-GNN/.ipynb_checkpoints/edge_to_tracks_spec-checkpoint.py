#!/usr/bin/env python
# coding: utf-8

# # Hits → Edge Classification → Tracks (spec-compliant)
# 
# This notebook converts **hits files** into **track files** using your trained AGNN edge classifier and connected components. The saved `.npz` outputs are compatible with your trigger TrackDataset (Nx15 `track_hits`, etc.) and are written to separate `trigger/` and `nontrigger/` directories.
# 
# ### Pipeline
# 1. Load `config.pkl` and weights from a training experiment folder.
# 2. For each event (`event*.npz`) in `input_dir` and `input_dir2`:
#    - Build the event graph using `datasets.hit_graph_trigger_pileup.load_graph`.
#    - Run the AGNN to classify edges and threshold probabilities.
#    - Compute connected components → **tracks**.
#    - Assemble **spec-compliant** fields and save a `.npz` named `event*.npz` in the appropriate output subfolder.
# 3. Show progress with `tqdm`.
# 

# In[1]:


# -------------------- User configuration --------------------
TRAIN_DIR = "/disks/disk4/giorgian/tracking_results/particle-model-tree/5xhcqcvt/agnn-lr0.001186351998699542-b12-d64-PReLU-gi1-ln-True-n50000/experiment_2025-09-13_05:54:17"  # contains config.pkl and weights
#TRAIN_DIR = "/disks/disk4/giorgian/old-trainings/particle-model-tree/biatt-augment-adj/agnn-lr0.001194944362880246-b12-d64-PReLU-gi3-ln-True-n50000/experiment_2025-09-22_00:30:00/"
TRAIN_DIR = "../tracking_results/agnn/agnn-lr0.0008981882248109734-b12-d64-ReLU-gi3-ln-True-n50000/experiment_2025-10-20_07:33:02" # D0, tracking, no pileup GNN
OUTPUT_ROOT =  "/disks/disk4/giorgian/tracking_inference/agnn/0okpolaz/cerulean-sweep-70"                    # will create 'trigger/' and 'nontrigger/'
#OUTPUT_ROOT = "/disks/disk4/giorgian/tracking_inference/agnn/b2excqac/distinctive-sweep-8"  

# Inference thresholds / knobs
PRED_THRESHOLD = 0.5  # sigmoid prob threshold for edge selection
MIN_TRACK_SIZE = 1     # set to 2 to drop isolated single-hit components

# Optional device (None → auto choose CUDA if available)
DEVICE = None  # e.g., "cuda:0" or "cpu"


# In[2]:


import os, sys, glob, pickle
import numpy as np
import torch
from tqdm.auto import tqdm
import random
# Prefer repo-style imports; fall back to local files if needed
try:
    from models.agnn import GNNSegmentClassifier
except Exception:
    from agnn import GNNSegmentClassifier

try:
    from datasets.hit_graph_trigger_pileup import load_graph as load_hits_graph
except Exception:
    # Local fallback
    from hit_graph_trigger_pileup import load_graph as load_hits_graph

try:
    import torch_geometric
    from torch_geometric.data import Data
except Exception as e:
    raise RuntimeError("torch_geometric is required for this notebook") from e

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "trigger"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "nontrigger"), exist_ok=True)

def pick_device(hint=None):
    if isinstance(hint, str):
        return torch.device(hint)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = pick_device(DEVICE)
DEVICE


# In[3]:


# -------------------- Config / model loading --------------------
def load_training_config(train_dir):
    with open(os.path.join(train_dir, "config.pkl"), "rb") as f:
        cfg = pickle.load(f)
    return cfg

def find_weights(train_dir):
    cands = []
    train_dir = os.path.join(train_dir, "checkpoints")
    for pat in ("*.pt", "*.pth", "*.tar"):
        cands += glob.glob(os.path.join(train_dir, pat))
        cands += glob.glob(os.path.join(train_dir, "**", pat), recursive=True)
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if not cands:
        raise FileNotFoundError(f"No checkpoint (*.pt|*.pth|*.tar) under {train_dir}")
    return cands[0]

def build_model(cfg):
    m = cfg.get("model", {})
    return GNNSegmentClassifier(
        input_dim=m.get("input_dim", 5),
        hidden_dim=m.get("hidden_dim", 64),
        n_graph_iters=m.get("n_graph_iters", 1),
        hidden_activation=m.get("hidden_activation", "PReLU"),
        layer_norm=m.get("layer_norm", True),
    )

def load_checkpoint(model, weights_path, map_location):
    ckpt = torch.load(weights_path, map_location=map_location)
    for key in ("state_dict", "model_state_dict", "model", "net"):
        if isinstance(ckpt, dict) and key in ckpt:
            model.load_state_dict(ckpt[key])
            return
    model.load_state_dict(ckpt)  # raw state_dict


# In[4]:


# -------------------- Geometry & helpers --------------------
import math

# Map physical layers -> 5-slot track representation
LAYER_GROUPS = [(0,), (1,), (2,), (3,4), (5,6)]
LAYER_TO_SLOT = {lid: i for i, group in enumerate(LAYER_GROUPS) for lid in group}

def calc_dphi(phi2, phi1):
    d = phi2 - phi1
    return (d + np.pi) % (2*np.pi) - np.pi

def compute_edge_features(hit_cyl, edge_index):
    r = hit_cyl[:, 0]; phi = hit_cyl[:, 1]; z = hit_cyl[:, 2]
    i0 = edge_index[0]; i1 = edge_index[1]
    dphi = calc_dphi(phi[i1], phi[i0])
    dr = (r[i1] - r[i0])
    dz = (z[i1] - z[i0])
    dr = np.where(dr == 0, 1e-6, dr)
    phi_slope = dphi / dr
    z0 = z[i0] - r[i0] * dz / dr
    return phi_slope.astype(np.float32), z0.astype(np.float32)

class DSU:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)
        self.r = np.zeros(n, dtype=np.int8)
    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def mode_ignore_nan(a):
    vals, counts = np.unique(a[~np.isnan(a)] if a.dtype.kind == 'f' else a, return_counts=True)
    return vals[np.argmax(counts)] if vals.size else (np.nan if a.dtype.kind == 'f' else -1)

def unwrap_phi(phi):
    if phi.size == 0: return phi
    ref = phi[0]
    return np.unwrap(phi - ref) + ref

def build_track_hits(hit_cart, layer_id, n_pixels, track_members):
    # Select 1 representative hit per 5-slot layer for each component
    N = len(track_members)
    track_hits = np.zeros((N, 15), dtype=np.float32)
    track_n_hits = np.zeros((N, 5), dtype=np.float32)
    n_pix_mat = np.zeros((N, 5), dtype=np.float32)
    for t, idx in enumerate(track_members):
        lids = layer_id[idx]
        pix = n_pixels[idx]
        for slot in range(5):
            mask = np.array([LAYER_TO_SLOT.get(int(l), -1) == slot for l in lids])
            track_n_hits[t, slot] = float(mask.sum())
            if not mask.any():
                continue
            cands = idx[mask]
            # choose hit with max n_pixels (deterministic)
            c_pix = n_pixels[cands]
            best = cands[int(np.argmax(c_pix))]
            xyz = hit_cart[best]
            track_hits[t, slot*3:(slot+1)*3] = xyz
            n_pix_mat[t, slot] = float(n_pixels[best])
    return track_hits, track_n_hits, n_pix_mat


# In[5]:


# -------------------- Main inference & save --------------------
def infer_and_save_tracks(train_dir, output_root, pthresh=0.5, min_track_size=1, device=None):
    device = torch.device(device) if isinstance(device, str) else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    cfg = load_training_config(train_dir)
    data_cfg = cfg.get("data", {})
    input_dir = os.path.expandvars(data_cfg.get("input_dir"))
    input_dir2 = os.path.expandvars(data_cfg.get("input_dir2", input_dir))
    use_intt = bool(data_cfg.get("use_intt", True))
    phi_slope_max = float(data_cfg.get("phi_slope_max", 0.03))
    z0_max = float(data_cfg.get("z0_max", 200.0))
    cyl_scale = tuple(data_cfg.get("cylindrical_features_scale", (3,1,3)))

    model = build_model(cfg).to(device)
    weights = find_weights(train_dir)
    load_checkpoint(model, weights, map_location=device)
    model.eval()

    def list_events(d):
        return sorted([os.path.join(d, f) for f in os.listdir(d) if f.startswith("event") and f.endswith(".npz")])

    files_trigger = list_events(input_dir)
    files_nontrig = list_events(input_dir2)
    all_files = [(f, True) for f in files_trigger] + [(f, False) for f in files_nontrig]
    random.shuffle(all_files)

    for path, is_trigger_dir in tqdm(all_files, desc="Events"):
        # Build event graph & meta (uses repo dataset util)
        x, edge_index, _, event_info = load_hits_graph(path, np.array(cyl_scale), phi_slope_max, z0_max, use_intt=use_intt, construct_edges=True)

        data = Data(x=torch.from_numpy(x).float(), edge_index=torch.from_numpy(edge_index).long()).to(device)
        with torch.no_grad():
            logits = model(data)
            prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        mask = (prob >= pthresh).astype(bool).reshape(-1)
        sel_edges = edge_index[:, mask]

        # Connected components on selected edges
        n_nodes = x.shape[0]
        dsu = DSU(n_nodes)
        if sel_edges.shape[1] > 0:
            e0 = sel_edges[0].astype(int); e1 = sel_edges[1].astype(int)
            for u, v in zip(e0, e1):
                dsu.union(u, v)
        parents = np.array([dsu.find(i) for i in range(n_nodes)], dtype=np.int64)
        uniq, inv = np.unique(parents, return_inverse=True)

        # Drop small components if requested
        comp_counts = np.bincount(inv, minlength=uniq.size)
        keep_comp = np.where(comp_counts >= min_track_size)[0]
        # members per kept component (list of index arrays)
        track_members = [np.where(inv == k)[0] for k in keep_comp]

        # Build track_hits (Nx15), per-layer hit counts (Nx5), per-layer n_pixels (Nx5)
        hit_cart = event_info.hit_cartesian
        hit_cyl = event_info.hit_cylindrical
        layer_id = event_info.layer_id
        n_pixels = event_info.n_pixels
        track_hits, track_n_hits, n_pix_mat = build_track_hits(hit_cart, layer_id, n_pixels, track_members)

        N = track_hits.shape[0]
        # Aggregate per-track attributes
        def agg_mean(a):
            return np.array([np.nanmean(a[idx], axis=0) if a.ndim == 2 else np.nanmean(a[idx]) for idx in track_members])

        # Per-track momentum (Nx3); energy (N,) using mean; origin (Nx3)
        momentum = agg_mean(event_info.momentum).astype(np.float32)
        energy = agg_mean(event_info.energy).astype(np.float32)
        track_origin = agg_mean(event_info.track_origin).astype(np.float32)

        # Modes for IDs/types; trigger_node flag per track (any)
        particle_id = np.array([mode_ignore_nan(event_info.particle_id[idx]) for idx in track_members]).astype(np.float32)
        particle_types = np.array([mode_ignore_nan(event_info.particle_type[idx]) for idx in track_members]).astype(np.float32)
        parent_particle_types = np.array([mode_ignore_nan(event_info.parent_particle_type[idx]) for idx in track_members]).astype(np.float32)
        trigger_node = np.array([np.any(event_info.trigger_node[idx]) for idx in track_members], dtype=np.float32)

        # Cylindrical std per track (r, phi, z) with phi unwrap
        cyl_std = np.zeros((N, 3), dtype=np.float32)
        for t, idx in enumerate(track_members):
            r = hit_cyl[idx, 0]
            phi = unwrap_phi(hit_cyl[idx, 1])
            z = hit_cyl[idx, 2]
            cyl_std[t] = np.array([np.std(r), np.std(phi), np.std(z)], dtype=np.float32)

        # Edge features & metadata
        edge_phi_slope, edge_z0 = compute_edge_features(hit_cyl, edge_index)

        # Event-level metadata
        interaction_point = event_info.interaction_point.astype(np.float32)
        trigger = np.array(event_info.trigger).astype(np.float32)
        has_trigger_pair = trigger.copy()  # placeholder if not available at hits-level

        # Save to trigger/nontrigger subdir; keep original base name (event*.npz)
        sub = "trigger" if (is_trigger_dir or bool(trigger)) else "nontrigger"
        out_dir = os.path.join(output_root, sub)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(path)  # e.g., eventXXXX.npz
        out_path = os.path.join(out_dir, base)

        payload = {
            # -------- Required by TrackDataset (tracks) --------
            "track_hits": track_hits,                      # (N, 15)
            "track_n_hits": track_n_hits.astype(int),                  # (N, 5) per-layer counts
            "n_pixels": n_pix_mat.astype(int),                         # (N, 5) per-layer pixels
            "energy": energy.reshape(-1),                  # (N,)
            "momentum": momentum,                          # (N, 3)
            "track_origin": track_origin,                  # (N, 3)
            "trigger_node": trigger_node.astype(bool),                  # (N,)
            "particle_id": particle_id.astype(int),                    # (N,)
            "particle_types": particle_types.astype(int),              # (N,)
            "parent_particle_types": parent_particle_types.astype(int),# (N,)
            "cylindrical_std": cyl_std,                    # (N, 3)
            # -------- Event-level --------
            "interaction_point": interaction_point,        # (3,)
            "trigger": trigger.astype(int),                            # scalar
            "has_trigger_pair": has_trigger_pair.astype(int),          # scalar
            # -------- Optional provenance (edges) --------
            "edge_index": edge_index.astype(np.int64),     # (2, E)
            "edge_confidence": prob.reshape(-1),           # (E,)
            "edge_mask": mask.astype(np.bool_),            # (E,)
            "edge_phi_slope": edge_phi_slope,              # (E,)
            "edge_z0": edge_z0,                            # (E,)
        }

        np.savez(out_path, **payload)

    return {
        "n_trigger": len(files_trigger),
        "n_nontrigger": len(files_nontrig),
        "output_root": output_root,
        "weights": weights,
    }

print("Configured. Set TRAIN_DIR/OUTPUT_ROOT above, then run infer_and_save_tracks(...)")


# In[6]:


import os
import numpy as np
import torch
from tqdm.auto import tqdm

# expects these to already exist (same as your old code):
# - load_training_config, build_model, find_weights, load_checkpoint
# - DSU, build_track_hits, mode_ignore_nan, unwrap_phi

def infer_and_save_tracks_with_dataloader(train_dir, output_root, pthresh=0.5, min_track_size=1, device=None):
    """
    Old infer-and-save (DSU + build_track_hits) but reading events from the SAME
    DataLoaders you used for training (whatever `data.name` is in your config).
    We force batch_size=1, ramp_up_nmix=False, load_full_event=True so that
    event_info is present and we still write the same NPZ schema as the old code.

    Output layout:
      {output_root}/train/{trigger,nontrigger}/eventXXXX.npz
      {output_root}/valid/{trigger,nontrigger}/eventXXXX.npz
    Returns split counts + paths.
    """
    # ---- device / config / model ----
    dev = torch.device(device) if isinstance(device, str) else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    cfg = load_training_config(train_dir)
    data_cfg = dict(cfg.get("data", {}))  # copy
    use_intt = bool(data_cfg.get("use_intt", True))

    model = build_model(cfg).to(dev)
    weights = find_weights(train_dir)
    load_checkpoint(model, weights, map_location=dev)
    model.eval()

    # ---- output dirs ----
    os.makedirs(output_root, exist_ok=True)
    for split in ("train", "valid"):
        os.makedirs(os.path.join(output_root, split, "trigger"), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, "nontrigger"), exist_ok=True)

    # ---- build DataLoaders exactly like training (honor data_cfg["name"]) ----
    # Only tweak a few knobs for inference:
    data_args = dict(data_cfg)
    data_args["batch_size"] = 1                 # 1 event per step
    data_args["ramp_up_nmix"] = False           # lock mixing at configured n_mix
    data_args["load_full_event"] = True         # ensure event_info is included
    data_args["n_train"] = int(300e3)
    # keep n_train/n_valid/n_workers/etc. from config (we won’t overwrite them)

    try:
        # Preferred path: same dispatcher training uses
        from datasets import get_data_loaders
        train_loader, valid_loader = get_data_loaders(distributed=False, **data_args)
    except Exception as e:
        # Fallback: build from get_datasets + pyg DataLoader
        from datasets import get_datasets
        from torch_geometric.loader import DataLoader as GeoDataLoader

        train_ds, valid_ds = get_datasets(
            n_train=data_args.get("n_train", 0),
            n_valid=data_args.get("n_valid", 0),
            input_dir=data_args.get("input_dir"),
            filelist=data_args.get("filelist"),
            real_weight=data_args.get("real_weight", 1.0),
            n_folders=data_args.get("n_folders", 1),
            input_dir2=data_args.get("input_dir2"),
            phi_slope_max=data_args.get("phi_slope_max", 0.03),
            z0_max=data_args.get("z0_max", 200.0),
            n_mix=data_args.get("n_mix", 1),
            use_intt=data_args.get("use_intt", True),
            load_full_event=True,               # critical for event_info
            load_all=False,
            construct_edges=data_args.get("construct_edges", True),
            drop_l1=data_args.get("drop_l1", False),
            drop_l2=data_args.get("drop_l2", False),
            drop_l3=data_args.get("drop_l3", False),
            intt_filter=data_args.get("intt_filter", False),
            add_global_node=data_args.get("add_global_node", False),
            ramp_up_nmix=False,
            ramp_rate=data_args.get("ramp_rate", 1),
            random_n_mix=data_args.get("random_n_mix", False),
            min_random_n_mix=data_args.get("min_random_n_mix", 1),
            trigger_edge_weight=data_args.get("trigger_edge_weight", 1),
        )
        n_workers = int(data_args.get("n_workers", 0))
        train_loader = GeoDataLoader(train_ds, batch_size=1, shuffle=False, num_workers=n_workers, pin_memory=True)
        valid_loader = GeoDataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=n_workers, pin_memory=True) if len(valid_ds) else None

    # ---- per-split processing (DSU → tracks → NPZ) ----
    def process_loader(loader, split_label):
        n_trig = n_non = 0
        if loader is None:
            return n_trig, n_non

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"{split_label} inference", unit="event")
        for idx, batch in pbar:
            if not hasattr(batch, "event_info"):
                # Should not happen if load_full_event=True, skip defensively
                continue

            with torch.no_grad():
                logits = model(batch.to(dev))
                prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32).reshape(-1)

            # graph primitives
            edge_index = batch.edge_index.detach().cpu().numpy()
            x = batch.x.detach().cpu().numpy()
            mask = (prob >= pthresh).astype(bool)
            sel_edges = edge_index[:, mask]

            # DSU over selected edges
            n_nodes = x.shape[0]
            dsu = DSU(n_nodes)
            if sel_edges.shape[1] > 0:
                e0 = sel_edges[0].astype(int); e1 = sel_edges[1].astype(int)
                for u, v in zip(e0, e1): dsu.union(u, v)
            parents = np.fromiter((dsu.find(i) for i in range(n_nodes)), dtype=np.int64, count=n_nodes)
            uniq, inv = np.unique(parents, return_inverse=True)
            comp_counts = np.bincount(inv, minlength=uniq.size)
            keep_comp = np.where(comp_counts >= int(min_track_size))[0]
            track_members = [np.where(inv == k)[0] for k in keep_comp]

            # event-level info (numpy views)
            ei = batch.event_info[0]
            hit_cart = np.asarray(ei.hit_cartesian)
            hit_cyl  = np.asarray(ei.hit_cylindrical)
            layer_id = np.asarray(ei.layer_id)
            n_pixels = np.asarray(ei.n_pixels)

            # track tensors
            track_hits, track_n_hits, n_pix_mat = build_track_hits(hit_cart, layer_id, n_pixels, track_members)
            N = track_hits.shape[0]

            # aggregate per-track attributes
            def agg_mean(a):
                arr = np.asarray(a)
                return np.array([np.nanmean(arr[idx], axis=0) if arr.ndim == 2 else np.nanmean(arr[idx]) for idx in track_members])

            momentum = agg_mean(ei.momentum).astype(np.float32)
            energy =   agg_mean(ei.energy).astype(np.float32)
            origin =   agg_mean(ei.track_origin).astype(np.float32)

            particle_id = np.array([mode_ignore_nan(np.asarray(ei.particle_id)[idx]) for idx in track_members]).astype(np.float32)
            particle_types = np.array([mode_ignore_nan(np.asarray(ei.particle_type)[idx]) for idx in track_members]).astype(np.float32)
            parent_particle_types = np.array([mode_ignore_nan(np.asarray(ei.parent_particle_type)[idx]) for idx in track_members]).astype(np.float32)
            trigger_node = np.array([np.any(np.asarray(ei.trigger_node)[idx]) for idx in track_members], dtype=np.float32)

            # cylindrical std per track (r, φ_unwrapped, z)
            cyl_std = np.zeros((N, 3), dtype=np.float32)
            for t, ids in enumerate(track_members):
                r = hit_cyl[ids, 0]
                phi = unwrap_phi(hit_cyl[ids, 1])
                z = hit_cyl[ids, 2]
                cyl_std[t] = np.array([np.std(r), np.std(phi), np.std(z)], dtype=np.float32)

            # optional embeddings
            track_embeddings = None
            if hasattr(model, "get_embedding"):
                with torch.no_grad():
                    emb = model.get_embedding(batch.to(dev)).detach().cpu().numpy().astype(np.float32)
                if emb.ndim == 2:
                    track_embeddings = emb[:N] if N <= emb.shape[0] \
                        else np.concatenate([emb, np.full((N - emb.shape[0], emb.shape[1]), np.nan, dtype=np.float32)], axis=0)

            # event-level flags & paths
            interaction_point = batch.interaction_point[0].detach().cpu().numpy().astype(np.float32)
            batch_trigger = bool(int(batch.trigger[0].detach().cpu().item())) if hasattr(batch, "trigger") else False
            sub_dir = "trigger" if batch_trigger else "nontrigger"
            n_trig += int(batch_trigger)
            n_non  += int(not batch_trigger)

            split_root = os.path.join(output_root, split_label, sub_dir)
            os.makedirs(split_root, exist_ok=True)
            base = (os.path.basename(batch.filename[0]) if hasattr(batch, "filename") and len(batch.filename)
                    else f"{split_label}_event_{idx:06d}.npz")
            out_path = os.path.join(split_root, base)

            # edge meta from dataset (already φ/z0-filtered there)
            edge_phi_slope = np.asarray(ei.edge_phi_slope)
            edge_z0        = np.asarray(ei.edge_z0)

            has_trigger_pair = np.asarray(ei.has_trigger_pair)
            if np.isscalar(has_trigger_pair) or (np.size(has_trigger_pair) == 1 and np.isnan(has_trigger_pair)):
                # fallback if missing
                has_trigger_pair = np.array(batch_trigger, dtype=int)

            payload = {
                # tracks
                "track_hits": track_hits,                          # (N, 15)
                "track_n_hits": track_n_hits.astype(int),          # (N, 5)
                "n_pixels": n_pix_mat.astype(int),                 # (N, 5)
                "energy": energy.reshape(-1),                      # (N,)
                "momentum": momentum,                              # (N, 3)
                "track_origin": origin,                            # (N, 3)
                "trigger_node": trigger_node.astype(bool),         # (N,)
                "particle_id": particle_id.astype(int),            # (N,)
                "particle_types": particle_types.astype(int),      # (N,)
                "parent_particle_types": parent_particle_types.astype(int),  # (N,)
                "cylindrical_std": cyl_std,                        # (N, 3)
                # event-level
                "interaction_point": interaction_point,            # (3,)
                "trigger": np.array(batch_trigger, dtype=int),     # scalar
                "has_trigger_pair": np.array(has_trigger_pair, dtype=int),
                # edges / provenance
                "edge_index": edge_index.astype(np.int64),         # (2, E)
                "edge_confidence": prob.reshape(-1),               # (E,)
                "edge_mask": mask.astype(bool),                    # (E,)
                "edge_phi_slope": edge_phi_slope,                  # (E,)
                "edge_z0": edge_z0,                                # (E,)
            }
            if track_embeddings is not None:
                payload["track_embeddings"] = track_embeddings

            np.savez(out_path, **payload)

            if (idx + 1) % 5 == 0 or (idx + 1) == len(loader):
                pbar.set_postfix(trigger=n_trig, nontrigger=n_non)

        return n_trig, n_non

    # ---- run both splits ----
    tr_trig, tr_non = process_loader(train_loader, "train")
    va_trig, va_non = process_loader(valid_loader, "valid")

    return {
        "train": {"n_trigger": tr_trig, "n_nontrigger": tr_non},
        "valid": {"n_trigger": va_trig, "n_nontrigger": va_non},
        "output_root": output_root,
        "weights": weights,
    }


# ### Example
# ```python
# res = infer_and_save_tracks(TRAIN_DIR, OUTPUT_ROOT, pthresh=PRED_THRESHOLD, min_track_size=MIN_TRACK_SIZE, device=str(DEVICE))
# res
# ```

# In[30]:


#infer_and_save_tracks(TRAIN_DIR, OUTPUT_ROOT, pthresh=PRED_THRESHOLD, min_track_size=MIN_TRACK_SIZE, device=str(DEVICE))


# In[ ]:


infer_and_save_tracks_with_dataloader(TRAIN_DIR, OUTPUT_ROOT, pthresh=0.5, min_track_size=1)


# In[ ]:




