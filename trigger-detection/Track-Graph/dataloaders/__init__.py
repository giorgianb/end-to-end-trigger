"""
PyTorch dataset specifications.
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

def get_data_loaders(name, batch_size, distributed=False,
                     n_workers=0, rank=None, n_ranks=None,
                     **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate

    if name == 'gt-trkgraph-nomasked':
        from . import gt_trkgraph_masked
        return gt_trkgraph_masked.get_data_loaders(name, batch_size, **data_args)
    elif name == 'pred-trackgraph-nomask':
        from . import pred_trackgraph
        return pred_trackgraph.get_data_loaders(name, batch_size, **data_args)
    else:
        raise Exception('Dataset %s unknown' % name)

