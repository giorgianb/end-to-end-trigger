"""
Python module for holding our PyTorch trainers.

Trainers here inherit from the BaseTrainer and implement the logic for
constructing the model as well as training and evaluation.
"""

def get_trainer(name, **trainer_args):
    """
    Factory function for retrieving a trainer.
    """
    if name == 'gnn_dense':
        from .gnn_dense import DenseGNNTrainer
        return DenseGNNTrainer(**trainer_args)
    elif name == 'gnn_sparse':
        from .gnn_sparse import SparseGNNTrainer
        return SparseGNNTrainer(**trainer_args)
    elif name == 'gnn_sparse_track':
        from .gnn_sparse_track import SparseGNNTrainer
        return SparseGNNTrainer(**trainer_args)
    elif name == 'gnn_trigger':
        from .gnn_trigger import SparseGNNTrainer
        return SparseGNNTrainer(**trainer_args)
    elif name == 'gnn_active':
        from .gnn_active import SparseGNNTrainer
        return SparseGNNTrainer(**trainer_args)
    elif name == 'gnn_ip':
        from .gnn_ip import SparseGNNTrainer
        return SparseGNNTrainer(**trainer_args)
    elif name == 'gnn_sparse_ip_trigger':
        from .gnn_sparse_ip_trigger import SparseGNNTrainer
        return SparseGNNTrainer(**trainer_args)
    elif name == 'gnn_sparse_momentum_trigger':
        from .gnn_sparse_momentum_trigger import SparseGNNTrainer
        return SparseGNNTrainer(**trainer_args)
    elif name == 'gnn_ip_2d':
        from .gnn_ip_2d import SparseGNNTrainer
        return SparseGNNTrainer(**trainer_args)
    elif name == 'gnn_trigger_contrastive':
        from .gnn_trigger_contrastive import SparseGNNTrainer
        return SparseGNNTrainer(**trainer_args)
    else:
        raise Exception('Trainer %s unknown' % name)
