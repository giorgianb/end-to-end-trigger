
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
        trigger_loss_func = nn.functional.binary_cross_entropy_with_logits
        momentum_loss_func = nn.functional.mse_loss
 
        # For trigger
        preds = []
        labels = []
        # For Momentum
        preds_momentum = []
        labels_momentum = []
 

        # Loop over training batches
        for i, batch in enumerate(tqdm.tqdm(data_loader, smoothing=0.0)):
            batch = batch.to(self.device)
            self.model.zero_grad()
            batch_output, momentum_output, trigger_output = self.model(batch)
            mask = batch_output == batch_output
            batch_loss = self.loss_func(batch_output[mask], batch.y.float()[mask], weight=batch.w[mask])
            #print(f'{momentum_output.shape=} {batch.momentum_intt.shape=}')
            momentum_loss = momentum_loss_func(momentum_output[:, :2], batch.momentum_intt[:, :2]) + 0.01 * momentum_loss_func(momentum_output[:, 2:], batch.momentum_intt[:, 2:])
            trigger_loss = trigger_loss_func(trigger_output, batch.trigger.float())
            batch_loss += trigger_loss + momentum_loss
            batch_loss.backward()
            self.optimizer.step()

            # For Trigger
            preds.append(trigger_output.cpu().data.numpy())
            trigger = batch.trigger.cpu().numpy()
            labels.append(trigger)


            # For Momentum
            preds_momentum.extend(momentum_output.cpu().data.numpy())
            momentum = batch.momentum_intt.cpu().numpy()
            labels_momentum.extend(momentum)
 
 
            sum_loss += batch_loss.item()

            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output[mask])
            matches = ((batch_pred > 0.5) == (batch.y[mask] > 0.5))
            edge_true = ((batch_pred > 0.5) & (batch.y[mask] > 0.5)).sum()
            #print(f'{(batch.y > 0.5).sum().item()/batch.y.shape[0]=} {(batch_pred > 0.5).sum().item()/batch.y.shape[0]=}')
            edge_count = (batch.y[mask] > 0.5).sum()
            nonedge_true = ((batch_pred < 0.5) & (batch.y[mask] < 0.5)).sum()
            nonedge_count = (batch.y[mask] < 0.5).sum()
            sum_correct += matches.sum().item()
            sum_edge_true += edge_true.item()
            sum_edge += edge_count.item()
            sum_nonedge_true += nonedge_true.item()
            sum_nonedge += nonedge_count.item()
            sum_total += matches.numel()


            # Dump additional debugging information
            if self.logger.isEnabledFor(logging.DEBUG):
                l1 = get_weight_norm(self.model, 1)
                l2 = get_weight_norm(self.model, 2)
                grad_norm = get_grad_norm(self.model)
                self.logger.debug('  train batch %i loss %.4f l1 %.2f l2 %.4f grad %.3f idx %i',
                                  i, batch_loss.item(), l1, l2, grad_norm, batch.i[0].item())

        # Summarize the epoch
        n_batches = i + 1
        labels_momentum = np.hstack(labels_momentum)
        preds_momentum = np.hstack(preds_momentum)
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
 
        result = {'trigger_prec': metrics.precision_score(labels, preds>0, average='macro'),
            'trigger_recall': metrics.recall_score(labels, preds>0, average='macro'),
            'trigger_acc': metrics.accuracy_score(labels, preds>0),
            'trigger_F1': metrics.f1_score(labels, preds>0, average="macro"),
            'explained_variance': metrics.explained_variance_score(labels_momentum, preds_momentum),
            'max_error': metrics.max_error(labels_momentum, preds_momentum),
            'rmse': np.sqrt(metrics.mean_squared_error(labels_momentum, preds_momentum)),
            'mae': metrics.mean_absolute_error(labels_momentum, preds_momentum),
            'r2': metrics.r2_score(labels_momentum, preds_momentum)
        }

 
        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / n_batches
        summary['train_acc'] = sum_correct / sum_total
        summary['recall'] = sum_edge_true / sum_edge
        summary['precision'] = sum_edge_true / (sum_edge_true + sum_nonedge - sum_nonedge_true) if sum_edge_true + sum_nonedge - sum_nonedge_true != 0 else 1
        summary['f1'] = 2 * summary['recall'] * summary['precision'] / (summary['recall'] + summary['precision'])
        summary['l1'] = get_weight_norm(self.model, 1)
        summary['l2'] = get_weight_norm(self.model, 2)
        summary['grad_norm'] = get_grad_norm(self.model)

        # For trigger
        summary['trigger_label_acc'] = metrics.accuracy_score(labels, preds>0)
        summary['trigger_auroc'] = metrics.roc_auc_score(labels, preds)


        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Model LR %f l1 %.2f l2 %.2f',
                          summary['lr'], summary['l1'], summary['l2'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        self.logger.info('  Precision: %.3f', summary['precision'])
        self.logger.info('  Recall: %.3f', summary['recall'])
        self.logger.info('  F1: %.3f', summary['f1'])
        self.logger.info(f'multi-learning result: {result}')

        if self.use_wandb:
            phase = 'train'
            wandb.log({phase.capitalize() + " Loss for tracking" : summary['train_loss']})
            wandb.log({phase.capitalize() + " Accuracy for tracking" : summary['train_acc']})
            wandb.log({phase.capitalize() + " L1 of parameters for tracking" : summary['l1']})
            wandb.log({phase.capitalize() + " L2 of parameters for tracking" : summary['l2']})
            wandb.log({phase.capitalize() + " Learning rate for tracking " : summary['lr']})
            wandb.log({phase.capitalize() + " Grad Norm for tracking " : summary['grad_norm']})
            wandb.log({phase.capitalize() + " Recall for tracking " : summary['recall']})
            wandb.log({phase.capitalize() + " Precision for tracking " : summary['precision']})
            wandb.log({phase.capitalize() + " F1 for tracking " : summary['f1']})
            wandb.log({phase.capitalize() + " Trigger Precison " : summary['trigger_label_acc']})
            wandb.log({phase.capitalize() + " Trigger Precison " : result['trigger_prec']})
            wandb.log({phase.capitalize() + " Trigger Recall " : result['trigger_recall']})
            wandb.log({phase.capitalize() + " Trigger F Score for " : result['trigger_F1']})
            wandb.log({phase.capitalize() + " Trigger Roc_auc for " : summary['trigger_auroc']})
            wandb.log({phase.capitalize() + " explained variance" : result['explained_variance']})
            wandb.log({phase.capitalize() + " max error" : result['max_error']})
            wandb.log({phase.capitalize() + " rmse" : result['rmse']})
            wandb.log({phase.capitalize() + " mae" : result['mae']})
            wandb.log({phase.capitalize() + " r2" : result['r2']})
 



        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        if not hasattr(self, 'best_validation_f1'):
            self.best_validation_f1 = 0

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
        trigger_loss_func = nn.functional.binary_cross_entropy_with_logits
        momentum_loss_func = nn.functional.mse_loss
 
        # For trigger
        preds = []
        labels = []
        # For Momentum
        preds_momentum = []
        labels_momentum = []
 
        # Loop over batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)

            # Make predictions on this batch
            batch_output, momentum_output, trigger_output = self.model(batch)
            mask = batch_output == batch_output
            batch_loss = self.loss_func(batch_output[mask], batch.y.float()[mask]).item()
            #print(f'{momentum_output.shape=} {batch.momentum_intt.shape=}')
            momentum_loss = momentum_loss_func(momentum_output[:, :2], batch.momentum_intt[:, :2]) + 0.01 * momentum_loss_func(momentum_output[:, 2:], batch.momentum_intt[:, 2:])
            trigger_loss = trigger_loss_func(trigger_output, batch.trigger.float())
            batch_loss += trigger_loss + momentum_loss

            # For Trigger
            preds.append((trigger_output).cpu().data.numpy())
            trigger = batch.trigger.cpu().numpy()
            labels.append(trigger)

            # For Momentum
            preds_momentum.extend(momentum_output.cpu().data.numpy())
            momentum = batch.momentum_intt.cpu().numpy()
            labels_momentum.extend(momentum)
 
            sum_loss += batch_loss

            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output[mask])
            matches = ((batch_pred > 0.5) == (batch.y[mask] > 0.5))
            edge_true = ((batch_pred > 0.5) & (batch.y[mask] > 0.5)).sum()
            edge_count = (batch.y[mask] > 0.5).sum()
            nonedge_true = ((batch_pred < 0.5) & (batch.y[mask] < 0.5)).sum()
            nonedge_count = (batch.y[mask] < 0.5).sum()
            sum_correct += matches.sum().item()
            sum_edge_true += edge_true.item()
            sum_edge += edge_count.item()
            sum_nonedge_true += nonedge_true.item()
            sum_nonedge += nonedge_count.item()
            sum_total += matches.numel()
            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)

        # Summarize the validation epoch
        n_batches = i + 1
        labels_momentum = np.hstack(labels_momentum)
        preds_momentum = np.hstack(preds_momentum)
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)

        result = {'trigger_prec': metrics.precision_score(labels, preds>0, average='macro'),
            'trigger_recall': metrics.recall_score(labels, preds>0, average='macro'),
            'trigger_acc': metrics.accuracy_score(labels, preds>0),
            'trigger_F1': metrics.f1_score(labels, preds>0, average="micro"),
            'explained_variance': metrics.explained_variance_score(labels_momentum, preds_momentum),
            'max_error': metrics.max_error(labels_momentum, preds_momentum),
            'rmse': np.sqrt(metrics.mean_squared_error(labels_momentum, preds_momentum)),
            'mae': metrics.mean_absolute_error(labels_momentum, preds_momentum),
            'r2': metrics.r2_score(labels_momentum, preds_momentum)
        }


        summary['valid_loss'] = sum_loss / n_batches
        summary['valid_acc'] = sum_correct / sum_total
        summary['valid_label_acc'] = summary['valid_acc']
        summary['recall'] = sum_edge_true / sum_edge
        summary['precision'] = sum_edge_true / (sum_edge_true + sum_nonedge - sum_nonedge_true) if (sum_edge_true + sum_nonedge - sum_nonedge_true) != 0 else 0
        #print(f'{sum_edge_true=} {sum_edge=} {sum_nonedge_true=} {sum_nonedge=}')
        f1 = 2 * summary['recall'] * summary['precision'] / (summary['recall'] + summary['precision']) if (summary['recall'] + summary['precision'] != 0) else 0
        summary['f1'] = f1
        # For trigger
        summary['trigger_label_acc'] = metrics.accuracy_score(labels, preds>0)
        summary['trigger_auroc'] = metrics.roc_auc_score(labels, preds)


        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), n_batches)
        self.logger.info('  Validation loss: %.3f acc: %.3f edge_true: %.3f edge: %.3f edge_acc: %.3f nonedge_true: %.3f nonedge: %.3f nonedge_acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc'], sum_edge_true, sum_edge, float(sum_edge_true)/float(sum_edge), sum_nonedge_true, sum_nonedge, float(sum_nonedge_true)/float(sum_nonedge)))
        self.logger.info(' Precision: %.3f', summary['precision'])
        self.logger.info(' Recall: %.3f', summary['recall'])
        self.logger.info(' F1: %.3f', summary['f1'])
        self.logger.info(f'multi-learning result: {result}')

        self.best_validation_f1 = max(self.best_validation_f1, summary['f1'])

        
        if self.use_wandb:
            phase = 'valid'
            wandb.log({phase.capitalize() + " Loss for tracking" : summary['valid_loss']})
            wandb.log({phase.capitalize() + " Accuracy for tracking" : summary['valid_acc']})
            wandb.log({phase.capitalize() + " Recall for tracking" : summary['recall']})
            wandb.log({phase.capitalize() + " Precision for tracking" : summary['precision']})
            wandb.log({phase.capitalize() + " F1 for tracking" : summary['f1']})
            wandb.log({phase.capitalize() + " Best F1 for tracking" : self.best_validation_f1})
            wandb.log({phase.capitalize() + " Trigger Accuracy " : summary['trigger_label_acc']})
            wandb.log({phase.capitalize() + " Trigger Precison " : result['trigger_prec']})
            wandb.log({phase.capitalize() + " Trigger Recall " : result['trigger_recall']})
            wandb.log({phase.capitalize() + " Trigger F Score for " : result['trigger_F1']})
            wandb.log({phase.capitalize() + " Trigger Roc_auc for " : summary['trigger_auroc']})
            wandb.log({phase.capitalize() + " explained variance" : result['explained_variance']})
            wandb.log({phase.capitalize() + " max error" : result['max_error']})
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
