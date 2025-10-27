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
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import torch_geometric

import tqdm

import wandb

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from dataloaders import get_data_loaders
from utils.log import write_checkpoint, load_config, load_checkpoint, config_logging, save_config, print_model_summary, get_terminal_columns, center_text, make_table, numeric_runtime
from utils.losses import SupCon

class ArgDict:
    pass

DEVICE = 'cuda:0'

def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default='configs/counterfactual.yaml')
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
    argparser.add_argument('--early_stopping_validity', type=float, default=0.65)
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


def train(data, vae_model, model, loss_params, data_config, optimizer, epoch, output_dir):
    train_info = do_epoch(data, vae_model, model, loss_params, data_config, epoch, optimizer=optimizer)
    write_checkpoint(checkpoint_id=epoch, model=vae_model, optimizer=optimizer, output_dir=output_dir)
    return train_info

def evaluate(data, vae_model, model, loss_params, data_config, epoch):
    with torch.no_grad():
        val_info = do_epoch(data, vae_model, model, loss_params, data_config, epoch, optimizer=None)
    return val_info

def do_epoch(data, vae_model, model, loss_params, data_config, epoch, optimizer=None):
    if optimizer is None:
        # validation epoch
        model.eval()
        vae_model.eval()
    else:
        # train epoch
        vae_model.train()

    start_time = datetime.now()

    # Iterate over batches
    accum_info = {k: 0.0 for k in (
        'loss', 
        'validity', 
        'feature_proximity',
        'set_proximity', 
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives',
        'loss_kl',
        'loss_ce',
        'loss_sm_feature',
        'loss_sm_set',
        'loss_kl_cf',
        'set_proximity',
        'feature_proximity'
    )}

    num_insts = 0
    non_empty = 0
    skipped_batches = 0
    preds = []
    preds_prob = []
    correct = []
    total_size = 0
    total_selected = 0
    alpha, beta = loss_params['alpha'], loss_params['beta']
    def pd(x, y):
        return torch.sqrt(torch.sum((x - y)**2, dim=-1))

    def cd(x, y):
        x_norm = torch.linalg.norm(x, dim=-1)
        y_norm = torch.linalg.norm(y, dim=-1)
        x_norm[x_norm == 0] = 1
        y_norm[y_norm == 0] = 1
        return torch.sum(x * y, dim=-1) / (x_norm * y_norm)

    for batch in tqdm.tqdm(data):
        tracks = batch.track_vector.to(DEVICE, torch.float)
        n_tracks = batch.n_tracks.to(DEVICE)
        trig = (batch.trigger.to(DEVICE) == 1).long()

        is_trigger_track = batch.is_trigger_track.to(DEVICE, torch.float).unsqueeze(-1)
        origin_vertices = batch.origin_vertices.to(DEVICE, torch.float)
        ip = batch.ip.to(DEVICE, torch.float).unsqueeze(1).repeat(1, tracks.shape[1], 1)
        ptypes = batch.ptypes.to(DEVICE, torch.float).unsqueeze(-1)
        energies = batch.energies.to(DEVICE, torch.float)
        momentums = batch.momentums.to(DEVICE, torch.float)

        mask = torch.zeros((tracks.shape[0], tracks.shape[1]))
        for i, n_track in enumerate(n_tracks):
            mask[i, :n_track] = 1
        mask = mask.to(DEVICE)


        gt_x = torch.cat([is_trigger_track, origin_vertices, ip, ptypes, energies, momentums], dim=-1)
        x_cf, z_mu, z_logvar, new_mask, z_u_mu, z_u_logvar = vae_model(tracks, gt_x, trig, mask)
        x_cf, _ = model.recalculate_geometric_features(x_cf, torch.zeros_like(x_cf).reshape(*x_cf.shape[:-1], 5, 3), **data_config)
        if epoch >= loss_params['drop_epoch_start']:
            pred = model(x_cf, new_mask)
        else:
            pred = model(x_cf, mask)
            new_mask = mask
        z_mu_cf, z_logvar_cf = vae_model.get_representation(x_cf, gt_x, trig, new_mask)
        loss = 0
        ce_loss = F.cross_entropy(pred, (1 -trig))
        print(f'{ce_loss=}')
        loss += alpha*ce_loss
        accum_info['loss_ce'] += ce_loss.item()
        loss_kl = torch.mean(0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1), dim=-1)
        loss_kl_batch = (loss_kl * mask).sum()/mask.sum()
        print(f'{loss_kl_batch=}')
        loss += loss_kl_batch
        accum_info['loss_kl'] += loss_kl_batch.item()
        loss_sm_feature = beta*(pd(tracks, x_cf)*mask).sum()/mask.sum()
        loss += loss_sm_feature
        print(f'{loss_sm_feature=}')
        accum_info['loss_sm_feature'] += loss_sm_feature.item()
        loss_sm_set = F.cross_entropy(new_mask.reshape(-1), mask.reshape(-1), reduction='none')/mask.sum()
        print(f'{loss_sm_set=}')
        loss += loss_sm_set
        accum_info['loss_sm_set'] += loss_sm_set.item()
        loss_kl_cf = torch.mean(0.5 * (((z_logvar_cf - z_logvar) + ((z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1), dim=-1)
        loss_kl_cf_batch = (loss_kl_cf * mask).sum()/mask.sum()
        loss += loss_kl_cf_batch
        print(f'{loss_kl_cf_batch=}')
        accum_info['loss_kl_cf'] += loss_kl_cf_batch.item()
        track_count = torch.sum(mask, dim=-1)
        non_empty += torch.sum(track_count > 0).item()
        track_count[track_count == 0] = 1
        accum_info['set_proximity'] += (torch.sum(((new_mask > 0.5) == (mask > 0.5))*mask, dim=-1)/track_count).sum().item()
        accum_info['feature_proximity'] += (torch.sum(cd(tracks, x_cf)*mask, dim=-1)/track_count).sum().item()

        batch_size = tracks.shape[0]

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accum_info = calc_metrics(1 - trig, pred, accum_info)
        accum_info['loss'] += loss.item()
        num_insts += batch_size

    tp = accum_info['true_positives']
    tn = accum_info['true_negatives']
    fp = accum_info['false_positives']
    fn = accum_info['false_negatives']

    if num_insts > 0:
        accum_info['loss'] /= num_insts
        accum_info['loss_ce'] /= num_insts
        accum_info['loss_kl'] /= num_insts
        accum_info['loss_kl_cf'] /= num_insts
        accum_info['loss_sm_feature'] /= num_insts
        accum_info['loss_sm_set'] /= num_insts
        accum_info['loss_kl_cf'] /= num_insts
        accum_info['set_proximity'] /= non_empty
        accum_info['feature_proximity'] /= num_insts
        accum_info['validity'] = (tp + tn) / (tp + tn + fp + fn)

           
    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]

    print('Skipped batches:', skipped_batches)


    return accum_info

def main():
     # Parse the command line
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    execute_training(args, config)

def execute_training(args, config):
    global DEVICE

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
    torch.cuda.set_device(int(args.gpu))
    DEVICE = 'cuda:' + str(args.gpu)

    name = config['wandb']['run_name'] + f'-experiment_{start_time:%Y-%m-%d_%H:%M:%S}'
    logging.info(name)

    if args.use_wandb and not args.skip_wandb_init:
        wandb.init(
            project=config['wandb']['project_name'],
            name=name,
            tags=config['wandb']['tags'],
            config=config
        )

    # Load data
    logging.info('Loading training, validation, and test data')
    dconfig = copy.copy(config['data'])

    train_data, val_data, test_data = get_data_loaders(**dconfig)
    logging.info('Loaded %g training samples', len(train_data.dataset))
    logging.info('Loaded %g validation samples', len(val_data.dataset))
    logging.info('Loaded %g test samples', len(test_data.dataset))

    mconfig = copy.copy(config['model'])
    from models.Bipartite_Attention_VAE import Bipartite_Attention_VAE as VAE_Model
    from models.Bipartite_Attention_Masked import Bipartite_Attention as Model

    vae_model = VAE_Model(
        **mconfig
    )
    vae_model = vae_model.to(DEVICE)

    with open(config['classifier_config'], 'rb') as handle:
        cconfig = pickle.load(handle)
    model_file_path = config['classifier_checkpoint_file_path']
    model = Model(
            **cconfig['model']
    )
    model = model.to(DEVICE)
    model = load_checkpoint(model_file_path, model)
    model.eval()

    # Optimizer
    oconfig = config['optimizer']
    params = vae_model.parameters()
    if config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(params=params,
                lr=oconfig['learning_rate'], 
                weight_decay=oconfig['weight_decay'], 
                betas=[oconfig['beta_1'], oconfig['beta_2']],
                eps=oconfig['eps']
        )
    elif oconfig['type'] == 'SGD':
        optimizer = torch.optim.SGD(params=params, lr=oconfig['learning_rate'], momentum=oconfig['momentum'], weight_decay=oconfig['weight_decay'])
    else:
        raise NotImplementedError(f'Optimizer {config["optimizer"]["type"]} not implemented.')


    decay_rate = oconfig["learning_rate_decay_rate"]
    def lr_schedule(epoch):
        return decay_rate**epoch

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print_model_summary(model)
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The number of model parameters is {num_params}')

    # Metrics
    train_loss = np.empty(config['epochs'], float)
    train_validity = np.empty(config['epochs'], float)
    val_loss = np.empty(config['epochs'], float)
    val_validity = np.empty(config['epochs'], float)

    best_epoch = -1
    best_val_validity = 0
    best_classifier_val_validity = 0
    best_model = None
    for epoch in range(1, config['epochs'] + 1):
        train_info = train(train_data, vae_model, model, config['loss'], config['data'], optimizer, epoch, config['output_dir'])
        table = make_table(
            ('Total loss', f"{train_info['loss']:.6f}"),
            ('Validity', f"{train_info['validity']:.6f}"),
            ('Feature Proximity', f"{train_info['feature_proximity']:.6f}"),
            ('Set Proximity', f"{train_info['set_proximity']:.6f}"),
            ('Runtime', f"{train_info['run_time']}")
        )

        logging.info('\n'.join((
            '',
            "#" * get_terminal_columns(),
            center_text(f"Training - {epoch:4}", ' '),
            table
        )))

        train_loss[epoch-1], train_validity[epoch-1] = train_info['loss'], train_info['validity']
        if args.use_wandb:
            wandb.log({"Train Loss" : train_info['loss']}, step=epoch)
            wandb.log({"Train KL Loss": train_info['loss_kl']}, step=epoch)
            wandb.log({"Train CE Loss": train_info['loss_ce']}, step=epoch)
            wandb.log({"Train KL CF Loss": train_info['loss_kl_cf']}, step=epoch)
            wandb.log({"Train KL Feature Similarity Loss": train_info['loss_sm_feature']}, step=epoch)
            wandb.log({"Train KL Set Similarity Loss": train_info['loss_sm_set']}, step=epoch)
            wandb.log({"Train Validity" : train_info['validity']}, step=epoch)
            wandb.log({"Train Feature Proximity" : train_info['feature_proximity']}, step=epoch)
            wandb.log({"Train Set Proximity" : train_info['set_proximity']}, step=epoch)
            wandb.log({"Train Run-Time": numeric_runtime(train_info['run_time'])}, step=epoch)


        val_info = evaluate(val_data, vae_model, model, config['loss'], config['data'], epoch)
        table = make_table(
            ('Total loss', f"{val_info['loss']:.6f}"),
            ('Validity', f"{val_info['validity']:.6f}"),
            ('Feature Proximity', f"{val_info['feature_proximity']:.6f}"),
            ('Set Proximity', f"{val_info['set_proximity']:.6f}"),
            ('Runtime', f"{val_info['run_time']}")
        )

        logging.info('\n'.join((
            '',
            center_text(f"Validation - {epoch:4}", ' '),
            table
            )))


        if val_info['validity'] > best_val_validity:
            best_val_validity = val_info['validity']
            best_epoch = epoch
            best_model = copy.deepcopy(vae_model)

        val_loss[epoch-1], val_validity[epoch-1] = val_info['loss'], val_info['validity']
        if args.use_wandb:
            wandb.log({"Validation Loss" : val_info['loss']}, step=epoch)
            wandb.log({"Validation KL Loss": val_info['loss_kl']}, step=epoch)
            wandb.log({"Validation CE Loss": val_info['loss_ce']}, step=epoch)
            wandb.log({"Validation KL CF Loss": val_info['loss_kl_cf']}, step=epoch)
            wandb.log({"Validation KL Feature Similarity Loss": val_info['loss_sm_feature']}, step=epoch)
            wandb.log({"Validation KL Set Similarity Loss": val_info['loss_sm_set']}, step=epoch)
            wandb.log({"Validation Validity" : val_info['validity']}, step=epoch)
            wandb.log({"Best Validation Validity" : best_val_validity}, step=epoch)
            wandb.log({"Validation Feature Proximity" : val_info['feature_proximity']}, step=epoch)
            wandb.log({"Validation Set Proximity" : val_info['set_proximity']}, step=epoch)
            wandb.log({"Validation Run-Time": numeric_runtime(val_info['run_time'])}, step=epoch)



        if args.early_stopping and epoch >= args.early_stopping_epoch and best_val_validity < args.early_stopping_validity:
            break

        lr_scheduler.step()
        
    
    del train_data, val_data


    logging.info(f'Best validation accuracy: {best_val_validity:.4f}, best epoch: {best_epoch}.')
    logging.info(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')

    test_info = evaluate(test_data, best_model, model, config['loss'], config['data'], config['epochs'] + 1)
    table = make_table(
            ('Total loss', f"{test_info['loss']:.6f}"),
            ('Validity', f"{test_info['validity']:.6f}"),
            ('Feature Proximity', f"{test_info['feature_proximity']:.6f}"),
            ('Set Proximity', f"{test_info['set_proximity']:.6f}"),
            ('Runtime', f"{test_info['run_time']}")

    )
    logging.info('\n'.join((
        '',
        center_text(f"Test", ' '),
        table
        )))


    if args.use_wandb:
        wandb.log({"Test Loss" : test_info['loss']}, step=config['epochs']+1)
        wandb.log({"Test Validity" : test_info['validity']}, step=config['epochs']+1)
        wandb.log({"Test Feature Proximity" : test_info['feature_proximity']}, step=config['epochs']+1)
        wandb.log({"Test Set Proximity" : test_info['set_proximity']}, step=config['epochs']+1)
        wandb.log({"Test Run-Time": numeric_runtime(test_info['run_time'])}, step=config['epochs']+1)



    # Saving to disk
    if args.save:
        output_dir = os.path.join(config['output_dir'], 'summary')
        i = 0
        while True:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)  # raises error if dir already exists
                break
            i += 1
            output_dir = output_dir[:-1] + str(i)
            if i > 9:
                logging.info(f'Cannot save results on disk. (tried to save as {output_dir})')
                return

        logging.info(f'Saving all to {output_dir}')
        torch.save(best_model.state_dict(), os.path.join(output_dir, "exp_model.pt"))
        shutil.copyfile(__file__, os.path.join(output_dir, 'main.py'))
        shutil.copytree('models/', os.path.join(output_dir, 'models/'))
        results_dict = {'train_loss': train_loss,
                        'train_validity': train_validity,
                        'val_loss': val_loss,
                        'val_validity': val_validity}
        df = pd.DataFrame(results_dict)
        df.index.name = 'epochs'
        df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        best_dict = {'best_val_validity': best_val_validity, 'best_epoch': best_epoch}
        best_df = pd.DataFrame(best_dict, index=[0])
        best_df.to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)

    logging.shutdown()



if __name__ == '__main__':
    main()
