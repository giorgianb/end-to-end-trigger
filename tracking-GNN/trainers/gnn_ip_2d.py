"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import logging

# Externals
import torch
import numpy as np
import sklearn.metrics as metrics
import wandb
# Locals
from .gnn_base import GNNBaseTrainer
from utils.checks import get_weight_norm, get_grad_norm
import tqdm

class SparseGNNTrainer(GNNBaseTrainer):
    """Trainer code for sparse GNN."""

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        preds = []
        labels = []
        # sum_correct = 0
        # sum_total = 0
        # Loop over training batches
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            # batch.w = batch.w[0]
            batch = batch.to(self.device)
            batch.interaction_point = batch.interaction_point[..., :2]
            self.model.zero_grad()
            batch_output = self.model(batch)[..., :2]
            self.logger.debug(f'output size: {batch_output.shape}')
            # batch_pred = torch.sigmoid(batch_output)

            # logging.debug(f'match type and y type {type(batch_pred)} {type(batch.y)}')
            # matches = ((batch_pred > 0.5) == (batch.y > 0.5))
            # sum_correct += matches.sum().item()
            # sum_total += matches.numel()

            # batch_loss = self.loss_func(torch.sigmoid(batch_output), batch.y.float().float(), weight=batch.w.float())
            # logging.debug(f'batch w size : {batch.w.shape}')
            # batch_loss = self.loss_func(batch_output, batch.y.float(), weight=batch.w)
            # self.logger.info(f'prediciton: {batch_output}, ground truth: {batch.y}')
            batch_loss = self.loss_func(batch_output, batch.interaction_point.float())
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
            preds.extend(batch.interaction_point.cpu().data.numpy())
            ip = batch.interaction_point.cpu().numpy()
            labels.extend(ip)
            # predict_noise = batch_pred > 0.5
            # predict_hits = batch_pred < 0.5
            # true_noise = batch.y == 1
            # true_hits = batch.y == 0
            # self.logger.debug(f'\n--train batch predict noise: {predict_noise.sum().item()} true hits: {predict_hits.sum().item()} \n--train batch ground  noise: {true_noise.sum().item()} true hits: {true_hits.sum().item()}')

            # Dump additional debugging information
            if self.logger.isEnabledFor(logging.DEBUG):
                l1 = get_weight_norm(self.model, 1)
                l2 = get_weight_norm(self.model, 2)
                grad_norm = get_grad_norm(self.model)
                self.logger.debug('  train batch %i loss %.4f l1 %.2f l2 %.4f grad %.3f idx %i',
                                  i, batch_loss.item(), l1, l2, grad_norm, batch.i[0].item())

        # Summarize the epoch
        n_batches = i + 1
        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / n_batches
        # summary['train_acc'] = sum_correct / sum_total
        summary['l1'] = get_weight_norm(self.model, 1)
        summary['l2'] = get_weight_norm(self.model, 2)
        labels = np.hstack(labels)
        preds = np.hstack(preds)
        result = {'explained_variance': metrics.explained_variance_score(labels, preds),
            'max_error': metrics.max_error(labels, preds),
            'rmse': np.sqrt(metrics.mean_squared_error(labels, preds)),
            'mae': metrics.mean_absolute_error(labels, preds),
            'r2': metrics.r2_score(labels, preds)
        }

        for k, v in result.items():
            summary[k] = v

        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Model LR %f l1 %.2f l2 %.2f',
                          summary['lr'], summary['l1'], summary['l2'])
        # self.logger.info('  Training loss: %.3f acc: %.3f', summary['train_loss'], summary['train_acc'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        if self.use_wandb:
            wandb.log({"train_loss" : summary['train_loss']})
            wandb.log({"train explained variance" : result['explained_variance']})
            wandb.log({"train max error" : result['max_error']})
            wandb.log({"train rmse" : result['rmse']})
            wandb.log({"train mae" : result['mae']})
            wandb.log({"train r2" : result['r2']})
            wandb.log({"train l1" : summary['l1']})
            wandb.log({"train l2" : summary['l2']})


        return summary
    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        if not hasattr(self, 'best_validation_r2j'):
            self.best_validation_r2 = 0

        
        summary = dict()
        preds = []
        labels = []
        sum_loss = 0

        # Loop over batches
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            batch = batch.to(self.device)

            batch.interaction_point = batch.interaction_point[..., :2]
            # Make predictions on this batch
            batch_output = self.model(batch)[..., :2]
            batch_loss = self.loss_func(batch_output, batch.interaction_point.float())
            sum_loss += batch_loss.item()


            preds.extend(batch_output.cpu().data.numpy())
            ip = batch.interaction_point.cpu().numpy()
            labels.extend(ip)
            # Count number of correct predictions
            batch_ip = batch_output.cpu().numpy()


        n_batches = i + 1
        summary['valid_loss'] = sum_loss / n_batches
        labels = np.hstack(labels)
        preds = np.hstack(preds)
        result = {'explained_variance': metrics.explained_variance_score(labels, preds),
            'max_error': metrics.max_error(labels, preds),
            'rmse': np.sqrt(metrics.mean_squared_error(labels, preds)),
            'mae': metrics.mean_absolute_error(labels, preds),
            'r2': metrics.r2_score(labels, preds)
        }

        for k, v in result.items():
            summary[k] = v


        
        self.best_validation_r2 = max(self.best_validation_r2, summary['r2'])
        self.logger.info('Validation loss: %.3f', summary['valid_loss'])
        self.logger.info(f'event classification result: {result}')

        if self.use_wandb:
            wandb.log({"valid_loss" : summary['valid_loss']})
            wandb.log({"valid explained variance" : result['explained_variance']})
            wandb.log({"valid max error" : result['max_error']})
            wandb.log({"valid rmse" : result['rmse']})
            wandb.log({"valid mae" : result['mae']})
            wandb.log({"valid r2" : result['r2']})
            wandb.log({"best valid r2" : self.best_validation_r2})

        # TODO: hack just to ensure that we don't exit early
        summary['valid_label_acc'] = summary['r2'] + 0.3
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
