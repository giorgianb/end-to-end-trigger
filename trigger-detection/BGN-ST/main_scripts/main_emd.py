import random
import os
import sys
import argparse
import copy
import shutil
import json
import logging
import yaml
import pickle
from pprint import pprint
from datetime import datetime
from functools import partial
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from icecream import ic
from collections import defaultdict
from scipy.optimize import linprog
import heapq
from functools import cmp_to_key

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import torch_geometric

import tqdm

import wandb

import gc

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from dataloaders import get_data_loaders
from utils.log import write_checkpoint, load_config, load_checkpoint, config_logging, save_config, print_model_summary, get_terminal_columns, center_text, make_table, numeric_runtime
from utils.losses import SupCon

DEVICE = 'cuda:0'

def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default='configs/emd.yaml')
    argparser.add_argument('-g', '--gpu', default='0', help='The gpu to run on')
    argparser.add_argument('--auto', action='store_true')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no_save', dest='save', action='store_false')
    argparser.set_defaults(save=True, debug_load=False)
    argparser.add_argument('-v', '--verbose', action='store_true')
    argparser.add_argument('--show-config', action='store_true')
    argparser.add_argument('--resume', action='store_true', default=0, help='Resume from last checkpoint')
    
    # Logging
    argparser.add_argument('--name', type=str, default=None,
            help="Run name")
    argparser.add_argument('--use_wandb', action='store_true',
                        help="use wandb project name")
    argparser.add_argument('--skip_wandb_init', action='store_true',
                        help="Skip wandb initialization (helpful if wandb was pre-initialized)")
    argparser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    argparser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")

    # Early Stopping
    argparser.add_argument('--early_stopping', action='store_true')
    argparser.set_defaults(early_stopping=False)
    argparser.add_argument('--early_stopping_accuracy', type=float, default=0.65)
    argparser.add_argument('--early_stopping_epoch', type=int, default=1)

    args = argparser.parse_args()

    return args

def calc_metrics(trig, pred, accum_info):
    with torch.no_grad():
        assert len(pred.shape) == 2
        pred = torch.softmax(pred, dim=-1)
        tp = torch.sum((trig == 1) * (torch.argmax(pred, dim=-1) == 1)).item()
        tn = torch.sum((trig == 0) * (torch.argmax(pred, dim=-1) == 0)).item()
        fp = torch.sum((trig == 0) * (torch.argmax(pred, dim=-1) == 1)).item()
        fn = torch.sum((trig == 1) * (torch.argmax(pred, dim=-1) == 0)).item()

        accum_info['true_positives'] += tp
        accum_info['true_negatives'] += tn
        accum_info['false_positives'] += fp
        accum_info['false_negatives'] += fn

    return accum_info

def emd(energies1, hits1, energies2, hits2):
    hits1 = hits1/np.expand_dims(np.linalg.norm(hits1, axis=-1), -1)
    hits2 = hits2/np.expand_dims(np.linalg.norm(hits2, axis=-1), -1)

    distances = np.einsum('ik,jk->ij', hits1, hits2)
    distances[distances < -1] = -1
    distances[distances > 1] = 1
    distances = np.arccos(distances)
    #print(f'{distances=}')

    e1 = np.sum(energies1)
    e2 = np.sum(energies2)

    e_min = min(e1, e2)
    energy_term = abs(e1 - e2)
    c = distances.reshape(-1)
    A_eq = np.ones((1, c.shape[0]))
    b_eq = e_min*np.ones(1)

    n1 = energies1.shape[0]
    n2 = energies2.shape[0]
    A_ub_1 = (np.expand_dims(np.arange(n1*n2)//n2, axis=0) == np.expand_dims(np.arange(n1), 1))
    b_ub_1 = energies1
    A_ub_2 = (np.expand_dims(np.arange(n1*n2) % n2, axis=0) == np.expand_dims(np.arange(n2), 1))
    b_ub_2 = energies2

    A_ub = np.concatenate([A_ub_1, A_ub_2], axis=0)
    b_ub = np.concatenate([b_ub_1, b_ub_2], axis=0)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    x = res.x.reshape(n1, n2)
    #print(f'{x=}')

    return res.fun, energy_term

class MinQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.energy_queue = []
        self.emd_queue = []

    def update(self, value, label, event_id):
        value = value + (label, event_id)
        if len(self.emd_queue) < self.max_size:
            heapq.heappush(self.emd_queue, cmp_to_key(lambda x, y: y[0] - x[0])(value))
        # in this case, we have something that's smaller than our largest element
        elif value[0] < self.queue[0][0]:
            # So we pop our largest element and push our smaller one
            heapq.heappushpop(self.emd_queue, cmp_to_key(lambda x, y: y[0] - x[0])(value))

        if len(self.energy_queue) < self.max_size:
            heapq.heappush(self.energy_queue, cmp_to_key(lambda x, y: y[1] - x[1])(value))
        # in this case, we have something that's smaller than our largest element
        elif value[1] < self.queue[0][1]:
            # So we pop our largest element and push our smaller one
            heapq.heappushpop(self.energy_queue, cmp_to_key(lambda x, y: y[1] - x[1])(value))

    def get_closest(self, R, n_neighbors):
        emd_queue = list(map(lambda x: x.obj, self.emd_queue))
        energy_queue = list(map(lambda x: x.obj, self.energy_queue))

        queue = np.concatenate([np.array(emd_queue), np.array(energy_queue)], axis=0)
        distances = queue[:, 0]/R + queue[:, 1]
        indices = np.argsort(distances)[:2*n_neighbors]
        #print(f'{distances[indices]=}')
        # Remove potentially duplicate events
        event_ids = queue[indices, 3]
        #print(f'{event_ids=}')
        _, unique_indices = np.unique(event_ids, return_index=True)
        #print(f'{unique_indices=}')
        indices = indices[unique_indices][:n_neighbors]
        #print(f'{indices=}')

        return queue[indices], distances[indices]

class KNeighborsClassifier:
    def __init__(self, n_closest):
        self.n_closest = n_closest


    def fit(self, train_energies, train_hits, train_labels, valid_energies, valid_hits, valid_labels):
        self.min_queues = []
        self.gt = np.array(valid_labels).squeeze(1)
        for event_energies, event_hits in tqdm.tqdm(zip(valid_energies, valid_hits), total=len(valid_energies)):
            event_distances = []
            min_queue = MinQueue(self.n_closest)
            # For this event, find the distance to each of our training points
            for i, (energies, hits, label) in enumerate(zip(train_energies, train_hits, train_labels)):
                # EMD returns (sum f_ij, |sum E_i - sum E_j|)
                e = emd(event_energies, event_hits, energies, hits)

                min_queue.update(e, label[0], i)
            self.min_queues.append(min_queue)

    def classify(self, mconfig):
        predictions = []
        for min_queue in self.min_queues:
            closest, distance = min_queue.get_closest(mconfig['R'], mconfig['n_neighbors'])
            if mconfig['weights'] == 'uniform':
                predictions.append(np.mean(closest[:, 2]))
            elif mconfig['weights'] == 'distance':
                weights = 1.0/distance
                predictions.append(np.sum(weights*closest[:, 2])/np.sum(weights))
        return np.array(predictions), self.gt

# Fit a KNN model on all of the training data
def fit_data(train_data, val_data, mconfig):
    # Initialize a KNN model from sklearn
    # Fit the model on the training data
    energies = []
    triggers = []
    tracks = []
    hits = []
    for i, batch in enumerate(tqdm.tqdm(train_data)):
        energy = np.array(batch.energies.reshape(-1).detach().cpu().numpy())
        if energy.shape[0] == 0:
            continue
        tracks = np.array(batch.track_vector.squeeze(0).detach().cpu().numpy())
        trigger = np.array(batch.trigger.detach().cpu().numpy())
        h1 = tracks[:, 0:3]
        h2 = tracks[:, 3:6]
        h3 = tracks[:, 6:9]
        h4 = tracks[:, 9:12]
        h5 = tracks[:, 12:15]
        # Guarantee that we have a valid hit
        empty = np.all(h1 == 0, -1)
        h1[empty] = h2[empty]
        empty = np.all(h1 == 0, -1)
        h1[empty] = h3[empty]
        empty = np.all(h1 == 0, -1)
        h1[empty] = h4[empty]
        empty = np.all(h1 == 0, -1)
        h1[empty] = h5[empty]
        empty = np.all(h1 == 0, -1)
        #assert np.sum(empty) == 0
        h1 = h1[~empty]
        energy = energy[~empty]

        if h1.shape[0] == 0:
            continue

        energies.append(energy)
        triggers.append(trigger)
        hits.append(h1)
        del batch

    train_energies = energies
    train_hits = hits
    train_triggers = triggers
    neighbors = KNeighborsClassifier(mconfig['n_closest'])

    energies = []
    triggers = []
    tracks = []
    hits = []
    for batch in tqdm.tqdm(val_data):
        energy = np.array(batch.energies.reshape(-1).detach().cpu().numpy())
        if energy.shape[0] == 0:
            continue
        tracks = np.array(batch.track_vector.squeeze(0).detach().cpu().numpy())
        trigger = np.array(batch.trigger.detach().cpu().numpy())

        h1 = tracks[:, 0:3]
        h2 = tracks[:, 3:6]
        h3 = tracks[:, 6:9]
        h4 = tracks[:, 9:12]
        h5 = tracks[:, 12:15]

        # Guarantee that we have a valid hit
        empty = np.all(h1 == 0, -1)
        h1[empty] = h2[empty]
        empty = np.all(h1 == 0, -1)
        h1[empty] = h3[empty]
        empty = np.all(h1 == 0, -1)
        h1[empty] = h4[empty]
        empty = np.all(h1 == 0, -1)
        h1[empty] = h5[empty]
        empty = np.all(h1 == 0, -1)
        #assert np.sum(empty) == 0
        h1 = h1[~empty]
        energy = energy[~empty]
        if h1.shape[0] == 0:
            continue


        energies.append(energy)
        triggers.append(trigger)
        hits.append(h1)

        del batch


    val_energies = energies
    val_hits = hits
    val_triggers = triggers

    neighbors.fit(train_energies, train_hits, train_triggers, val_energies, val_hits, val_triggers)

    return neighbors


def evaluate(model, mconfig):
    accum_info = {k: 0.0 for k in (
        'ri', 
        'auroc', 
        'fscore', 
        'precision', 
        'recall', 
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives'
    )}

    num_insts = 0
    skipped_batches = 0
    preds_prob, correct = model.classify(mconfig)
    preds = preds_prob > 0.5

    tn = np.sum(~preds[correct == 0])
    tp = np.sum(preds[correct == 1])
    fp = np.sum(preds[correct == 0])
    fn = np.sum(~preds[correct == 1])

    accum_info['ri'] = (tp + tn)/(tp + tn + fp + fn)
    accum_info['precision'] = tp / (tp + fp) if tp + fp != 0 else 0
    accum_info['recall'] = tp / (tp + fn) if tp + fn != 0 else 0
    accum_info['fscore'] = (2 * tp)/(2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
    accum_info['true_positives'] = tp
    accum_info['true_negatives'] = tn
    accum_info['false_positives'] = fp
    accum_info['false_negatives'] = fn

    try:
        accum_info['auroc'] = roc_auc_score(correct, preds_prob)
    except ValueError:
        accum_info['auroc'] = 0
           

    print('Skipped batches:', skipped_batches)

    return accum_info

def main():
     # Parse the command line
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    dconfig = copy.copy(config['data'])

    train_data, val_data, test_data = get_data_loaders(**dconfig)
    model = fit_data(train_data, val_data, config['model'])

    execute_evaluation(model, args, config)

def test():
    # Load configuration
    config = load_config('configs/emd.yaml')
    dconfig = copy.copy(config['data'])

    train_data, val_data, test_data = get_data_loaders(**dconfig)
    model = fit_data(train_data, val_data, config['model'])

    return train_data, val_data, test_data, model


def execute_evaluation(model, args, config):
    start_time = datetime.now()
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    config['output_dir'] = os.path.join(config['output_dir'], f'experiment_{start_time:%Y-%m-%d_%H:%M:%S}')
    os.makedirs(config['output_dir'], exist_ok=True)

    # Setup logging
    file_handler = config_logging(verbose=args.verbose, output_dir=config['output_dir'],
                   append=args.resume, rank=0)

    logging.info('Command line config: %s' % args)
    logging.info('Configuration: %s', config)
    logging.info('Saving job outputs to %s', config['output_dir'])

    # Save configuration in the outptut directory
    save_config(config)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    name = config['wandb']['run_name'] + f'-experiment_{start_time:%Y-%m-%d_%H:%M:%S}'
    logging.info(name)

    if args.use_wandb and not args.skip_wandb_init:
        wandb.init(
            project=config['wandb']['project_name'],
            name=name,
            tags=config['wandb']['tags'],
            config=config
        )


    mconfig = copy.copy(config['model'])
    # Metrics

    val_info = evaluate(model, mconfig)
    table = make_table(
        ('Rand Index', f"{val_info['ri']:.6f}"),
        ('F-score', f"{val_info['fscore']:.4f}"),
        ('Recall', f"{val_info['recall']:.4f}"),
        ('Precision', f"{val_info['precision']:.4f}"),
        ('True Positives', f"{val_info['true_positives']}"),
        ('False Positives', f"{val_info['false_positives']}"),
        ('True Negatives', f"{val_info['true_negatives']}"),
        ('False Negatives', f"{val_info['false_negatives']}"),
        ('AUC Score', f"{val_info['auroc']:.6f}"),
    )

    logging.info('\n'.join((
        '',
        center_text(f"Validation", ' '),
        table
        )))

    best_val_ri = val_info['ri']
    best_val_auroc = val_info['auroc']

    if args.use_wandb:
        wandb.log({"Validation Accuracy" : val_info['ri']}, step=0)
        wandb.log({"Validation Precision" : val_info['precision']}, step=0)
        wandb.log({"Validation Recall": val_info['recall']}, step=0)
        wandb.log({"Validation F-Score": val_info['fscore']}, step=0)
        wandb.log({"Validation AUROC": val_info['auroc']}, step=0)
        wandb.log({"Best Validation Accuracy": best_val_ri}, step=0)


    logging.info(f'Validation accuracy: {best_val_ri:.4f}.')
    logging.info(f'Evaluation runtime: {str(datetime.now() - start_time).split(".")[0]}')


    logging.shutdown()



if __name__ == '__main__':
    main()
