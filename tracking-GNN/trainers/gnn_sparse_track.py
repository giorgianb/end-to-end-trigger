"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import logging

# Externals
import torch
import wandb

# Locals
from .gnn_base import GNNBaseTrainer
from utils.checks import get_weight_norm, get_grad_norm
import tqdm
import torch.nn as nn
import numpy as np

import sklearn.metrics as metrics

class SparseGNNTrainer(GNNBaseTrainer):
    """Trainer code for sparse GNN."""

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0
        sum_edge_true = 0
        sum_edge = 0
        sum_nonedge_true = 0
        sum_nonedge = 0
        sigmoid = torch.nn.Sigmoid()
        track_loss_func = nn.functional.mse_loss
 
        # For IP
        preds_intt_tracks = []
        labels_intt_tracks = []
 

        # Loop over training batches
        for i, batch in enumerate(tqdm.tqdm(data_loader, smoothing=0.0)):
            batch = batch.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch)
            track_loss = track_loss_func(batch_output, batch.intt_tracks.to(torch.float))
            batch_loss = track_loss
            batch_loss.backward()
            self.optimizer.step()


            # For IP
            preds_intt_tracks.append(batch_output.cpu().data.numpy())
            intt_tracks = batch.intt_tracks.cpu().numpy()
            labels_intt_tracks.append(intt_tracks)
 
 
            sum_loss += batch_loss.item()
            sum_total += batch.trigger.numel()

        n_batches = i + 1
        labels_intt_tracks = np.concatenate(labels_intt_tracks, axis=0)
        preds_intt_tracks = np.concatenate(preds_intt_tracks, axis=0)

        result = {            
                'explained_variance': metrics.explained_variance_score(labels_intt_tracks, preds_intt_tracks),
            'rmse': np.sqrt(metrics.mean_squared_error(labels_intt_tracks, preds_intt_tracks)),
            'mae': metrics.mean_absolute_error(labels_intt_tracks, preds_intt_tracks),
            'r2': metrics.r2_score(labels_intt_tracks, preds_intt_tracks)
        }

 
        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / n_batches
        summary['l1'] = get_weight_norm(self.model, 1)
        summary['l2'] = get_weight_norm(self.model, 2)


        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Model LR %f l1 %.2f l2 %.2f',
                          summary['lr'], summary['l1'], summary['l2'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        self.logger.info('  explained variance: %.3f', result['explained_variance'])
        self.logger.info(f'multi-learning result: {result}')

        if self.use_wandb:
            phase = 'train'
            wandb.log({phase.capitalize() + " Loss for tracking" : summary['train_loss']})
            wandb.log({phase.capitalize() + " L1 of parameters for tracking" : summary['l1']})
            wandb.log({phase.capitalize() + " L2 of parameters for tracking" : summary['l2']})
            wandb.log({phase.capitalize() + " Learning rate for tracking " : summary['lr']})
            wandb.log({phase.capitalize() + " explained variance for tracking " : result['explained_variance']})
            wandb.log({phase.capitalize() + " rmse" : result['rmse']})
            wandb.log({phase.capitalize() + " mae" : result['mae']})
            wandb.log({phase.capitalize() + " r2" : result['r2']})
 



        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        if not hasattr(self, 'best_explained_variance'):
            self.best_explained_variance = 0

        # Prepare summary information
        summary = dict()
        sum_total = 0
        sum_loss = 0
        sigmoid = torch.nn.Sigmoid()
        trigger_loss_func = nn.functional.binary_cross_entropy_with_logits
        track_loss_func = nn.functional.mse_loss
 
        # For IP
        preds_intt_tracks = []
        labels_intt_tracks = []
 
        # Loop over batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            batch_output = self.model(batch)
            track_loss = track_loss_func(batch_output, batch.intt_tracks)
            batch_loss = track_loss
 

            # For IP
            preds_intt_tracks.append(batch_output.cpu().data.numpy())
            intt_tracks = batch.intt_tracks.cpu().numpy()
            labels_intt_tracks.append(intt_tracks)
 
            sum_loss += batch_loss

            sum_total += batch.trigger.numel()
            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)

        # Summarize the validation epoch
        n_batches = i + 1
        labels_intt_tracks = np.concatenate(labels_intt_tracks, axis=0)
        preds_intt_tracks = np.concatenate(preds_intt_tracks, axis=0)

        result = {  
                'explained_variance': metrics.explained_variance_score(labels_intt_tracks, preds_intt_tracks),
            'rmse': np.sqrt(metrics.mean_squared_error(labels_intt_tracks, preds_intt_tracks)),
            'mae': metrics.mean_absolute_error(labels_intt_tracks, preds_intt_tracks),
            'r2': metrics.r2_score(labels_intt_tracks, preds_intt_tracks)
        }


        summary['valid_loss'] = sum_loss / n_batches
        #print(f'{sum_edge_true=} {sum_edge=} {sum_nonedge_true=} {sum_nonedge=}')

        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), n_batches)
        self.logger.info('  Validation loss: %.3f' %
                         (summary['valid_loss']))
        self.logger.info(f'multi-learning result: {result}')

        self.best_explained_variance = max(self.best_explained_variance, result['explained_variance'])

        
        if self.use_wandb:
            phase = 'valid'
            wandb.log({phase.capitalize() + " Loss for tracking" : summary['valid_loss']})
            wandb.log({phase.capitalize() + " explained variance for tracking" : result['explained_variance']})
            wandb.log({phase.capitalize() + " Best explained variance for tracking" : self.best_explained_variance})
            wandb.log({phase.capitalize() + " rmse" : result['rmse']})
            wandb.log({phase.capitalize() + " mae" : result['mae']})
            wandb.log({phase.capitalize() + " r2" : result['r2']})




        return summary

    @torch.no_grad()
    def predict(self, data_loader):
        preds, targets = [], []
        for batch in data_loader:
            preds.append(torch.sigmoid(self.model(batch)).squeeze(0))
            targets.append(batch.y.squeeze(0))
        return preds, targets

def _test():
    t = SparseGNNTrainer(output_dir='./')
    t.build_model()
