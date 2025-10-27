#!/usr/bin/env python3
"""
Per-sample inference dumper.

Usage:
  python infer.py /path/to/run_dir \
      --split train \
      --device auto \
      --gpu 0 \
      --outdir None

- /path/to/run_dir can be either:
  * the exact experiment folder that contains config.pkl and checkpoints/, or
  * a parent folder containing multiple experiment_* subfolders (the most recent is chosen).

Outputs:
  <run_dir>/inference_<timestamp>/sample_XXXXXX.npz (one per graph)
  Each NPZ includes:
    x, edge_index, (optional) edge_attr, pos, w, i, y or trigger,
    logits, probs, pred
"""

import os
import sys
import re
import glob
import argparse
import logging
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports from your project
from datasets import get_data_loaders
from trainers import get_trainer


# ------------------------- CLI & logging -------------------------

def parse_args():
    p = argparse.ArgumentParser("infer.py")
    p.add_argument("run_dir", help="Experiment folder OR parent containing experiment_*")
    p.add_argument("--split", choices=["train", "valid"], default="train",
                   help="Which split to run inference on (default: train)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Device selection (default: auto)")
    p.add_argument("--gpu", type=int, default=0, help="GPU index if using CUDA (default: 0)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Override data.num_workers (None keeps config)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Force batch size (default: 1 for per-sample files)")
    p.add_argument("--outdir", default=None,
                   help="Override output directory for npz files; default=run_dir/inference_<ts>")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    return p.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


# ------------------------- Helpers -------------------------

def resolve_experiment_dir(run_dir: str) -> str:
    """
    Accept either:
      - an experiment directory with config.pkl; or
      - a parent containing experiment_* subdirs (choose newest).
    """
    cfg_here = os.path.join(run_dir, "config.pkl")
    if os.path.isfile(cfg_here):
        return run_dir

    # Look for experiment_* subfolders and choose the most recent by natural sort
    candidates = [d for d in glob.glob(os.path.join(run_dir, "experiment_*")) if os.path.isdir(d)]
    if not candidates:
        raise FileNotFoundError(f"No config.pkl and no experiment_* subfolders in: {run_dir}")

    # Sort by timestamp in the folder name if present; fall back to mtime
    def sort_key(path):
        m = re.search(r"experiment_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})", os.path.basename(path))
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y-%m-%d_%H:%M:%S")
            except Exception:
                pass
        return datetime.fromtimestamp(os.path.getmtime(path))

    candidates.sort(key=sort_key)
    chosen = candidates[-1]
    logging.info("Resolved experiment dir: %s", chosen)
    return chosen


def find_latest_checkpoint(exp_dir: str) -> str:
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Missing checkpoints/ in {exp_dir}")
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("model_checkpoint")]
    if not files:
        raise FileNotFoundError(f"No model_checkpoint* files in {ckpt_dir}")
    files.sort()
    ckpt = os.path.join(ckpt_dir, files[-1])
    logging.info("Using checkpoint: %s", ckpt)
    return ckpt


def choose_device(arg_device: str, gpu_index: int) -> torch.device:
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_index}")
        logging.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


def to_numpy_safe(obj):
    """Convert a tensor-like to numpy if possible; otherwise return None."""
    try:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        if isinstance(obj, (np.ndarray, np.number, list, tuple, int, float, bool)):
            return np.array(obj)
    except Exception:
        pass
    return None


def data_to_numpy_dict(data_obj):
    """
    Extract common inputs/labels from a PyG Data object into numpy arrays.
    Saves only keys that exist.
    """
    keys_to_try = [
        "x", "x_intt", "x_mvtx", "edge_index", "edge_index_intt", "edge_attr", "pos", "w", "i", "y", "trigger"
    ]
    out = {}
    for k in keys_to_try:
        if hasattr(data_obj, k):
            arr = to_numpy_safe(getattr(data_obj, k))
            if arr is not None:
                out[k] = arr
    return out


# ------------------------- Main Inference -------------------------

def main():
    args = parse_args()
    setup_logging(args.verbose)

    # Resolve actual experiment dir (contains config.pkl & checkpoints/)
    exp_dir = resolve_experiment_dir(args.run_dir)

    # Load saved config
    cfg_path = os.path.join(exp_dir, "config.pkl")
    logging.info("Loading config: %s", cfg_path)
    with open(cfg_path, "rb") as f:
        config = pickle.load(f)

    # Prepare output dir for NPZs
    if args.outdir is not None and args.outdir.lower() != "none":
        out_dir = args.outdir
    else:
        out_dir = os.path.join(exp_dir, f"inference_{datetime.now():%Y-%m-%d_%H-%M-%S}")
    os.makedirs(out_dir, exist_ok=True)
    logging.info("Writing NPZs to: %s", out_dir)

    # Device
    device = choose_device(args.device, args.gpu)
    logging.info("Using device: %s", device)

    # Build trainer & model (reusing your existing factory)
    trainer = get_trainer(
        distributed_mode=None,
        output_dir=exp_dir,
        rank=0,
        n_ranks=1,
        gpu=(device.index if device.type == "cuda" else None),
        use_wandb=False,
        **config.get("trainer", {})
    )

    model_cfg = dict(config.get("model", {}))  # copy
    opt_cfg = config.get("optimizer", {})
    # Build model; this uses your config['model']['name'] and sets trainer.model
    trainer.build_model(optimizer_config=opt_cfg, **model_cfg)
    trainer.model.to(device)
    trainer.model.eval()

    # Load latest checkpoint weights
    ckpt_file = find_latest_checkpoint(exp_dir)
    ckpt = torch.load(ckpt_file, map_location=device)
    trainer.model.load_state_dict(ckpt["model"])
    logging.info("Checkpoint loaded successfully.")

    # Build data loaders; force batch_size=1 for per-sample NPZ
    data_cfg = dict(config.get("data", {}))  # copy
    data_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["n_workers"] = args.num_workers

    train_loader, valid_loader = get_data_loaders(
        distributed=False, rank=0, n_ranks=1, **data_cfg
    )
    loader = train_loader if args.split == "train" else valid_loader
    if loader is None:
        raise RuntimeError(f"No data loader for split '{args.split}'")

    logging.info("Processing %d samples from the %s split (batch_size=%d)",
                 len(loader.dataset), args.split, args.batch_size)

    # Inference loop
    n_saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Infer({args.split})"):
            # Keep the Batch intact so .batch exists (batch_size=1 recommended)
            data_obj = batch.to(device)
    
            # create a single-graph batch vector.
            if not hasattr(data_obj, "batch") or data_obj.batch is None:
                if hasattr(data_obj, "x") and data_obj.x is not None:
                    data_obj.batch = torch.zeros(
                        data_obj.x.size(0), dtype=torch.long, device=data_obj.x.device
                    )

            if not hasattr(data_obj, "batch_intt") and hasattr(data_obj, "x_intt"):
                data_obj.batch_intt = torch.zeros(
                    data_obj.x_intt.size(0), dtype=torch.long, device=data_obj.x_intt.device
                )
            if not hasattr(data_obj, "batch_mvtx") and hasattr(data_obj, "x_mvtx"):
                data_obj.batch_mvtx = torch.zeros(
                    data_obj.x_mvtx.size(0), dtype=torch.long, device=data_obj.x_mvtx.device
                )

            # Forward pass -> logits, probs, pred (threshold at logit > 0 to match training)
            logits = trainer.model(data_obj)
            probs = torch.sigmoid(logits)
            pred = (logits > 0).to(torch.int64)
    
            # Collect inputs/labels to numpy
            payload = data_to_numpy_dict(data_obj)
    
            # Choose a sample_id if available, otherwise use running counter
            sample_id = n_saved

    
            # Add outputs
            payload["logits"] = logits.detach().cpu().numpy()
            payload["probs"]  = probs.detach().cpu().numpy()
            payload["pred"]   = pred.detach().cpu().numpy()
    
            # Save one NPZ per sample
            out_path = os.path.join(out_dir, f"sample_{sample_id:06d}.npz")
            np.savez_compressed(out_path, **payload)
            n_saved += 1

    logging.info("Saved %d NPZ files to %s", n_saved, out_dir)


if __name__ == "__main__":
    main()
