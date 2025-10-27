"""
PyTorch dataset specifications.
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
import torch
import torch_geometric

def get_data_loaders(name, batch_size, distributed=False,
                     n_workers=0, rank=None, n_ranks=None,
                     **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate
    if name == 'hit_graph':
        from torch_geometric.data import Batch
        from . import hit_graph
        train_dataset, valid_dataset = hit_graph.get_datasets(**data_args)
        collate_fn = Batch.from_data_list
    elif name == 'hit_graph_trigger':
        from torch_geometric.data import Batch
        from . import hit_graph_trigger
        train_dataset, valid_dataset = hit_graph_trigger.get_datasets(**data_args)

        collate_fn = Batch.from_data_list
    elif name == 'hit_graph_trigger_real_data':
        from torch_geometric.data import Batch
        from . import hit_graph_trigger_real_data
        train_dataset, valid_dataset = hit_graph_trigger_real_data.get_datasets(**data_args)

        collate_fn = Batch.from_data_list
    elif name == 'hit_graph_trigger_pileup_bp':
        from torch_geometric.data import Batch
        from . import hit_graph_trigger_pileup_bp
        train_dataset, valid_dataset = hit_graph_trigger_pileup_bp.get_datasets(**data_args)
        collate_fn = Batch.from_data_list
    elif name == 'hit_graph_bp':
        from torch_geometric.data import Batch
        from . import hit_graph_bp
        train_dataset, valid_dataset = hit_graph_bp.get_datasets(**data_args)
        def collate_batch(base_data):
            x_intt = torch.cat([data.x_intt for data in base_data], dim=0)
            x_mvtx = torch.cat([data.x_mvtx for data in base_data], dim=0)
            momentum_intt = torch.cat([data.momentum_intt for data in base_data], dim=0)
            intt_tracks = torch.cat([data.intt_tracks for data in base_data], dim=0)
            y = torch.cat([data.y for data in base_data], dim=0)
            trigger = torch.tensor([data.trigger for data in base_data])
            w = torch.cat([data.w for data in base_data], dim=0)
            trigger_node = torch.cat([data.trigger_node for data in base_data], dim=0)
            active_node = torch.cat([data.active_node for data in base_data], dim=0)
            particle_id_intt = torch.cat([data.particle_id_intt for data in base_data], dim=0)
            particle_id_mvtx = torch.cat([data.particle_id_mvtx for data in base_data], dim=0)

            i = torch.tensor([data.i for data in base_data])
            filename = [data.filename for data in base_data]
            interaction_point = torch.cat([data.interaction_point for data in base_data], dim=0)
            if hasattr(base_data[0], 'event_info'):
                event_info = [data.event_info for data in base_data]

            edge_indices = []
            edge_indices_intt = []
            batch_intt = []
            batch_mvtx = []
            mvtx_counter = 0
            intt_counter = 0
            for i, data in enumerate(base_data):
                edge_index = data.edge_index
                edge_index_intt = data.edge_index_intt
                edge_index[0] += intt_counter
                edge_index[1] += mvtx_counter
                edge_index_intt[0] += intt_counter
                edge_index_intt[1] += intt_counter
                batch_intt.append(i*torch.ones(data.x_intt.size(0), dtype=int))
                batch_mvtx.append(i*torch.ones(data.x_mvtx.size(0), dtype=int))

                edge_indices.append(edge_index)
                edge_indices_intt.append(edge_index_intt)
                mvtx_counter += data.x_mvtx.size(0)
                intt_counter += data.x_intt.size(0)
            edge_index = torch.cat(edge_indices, dim=1)
            edge_index_intt = torch.cat(edge_indices_intt, dim=1)
            batch_intt = torch.cat(batch_intt)
            batch_mvtx = torch.cat(batch_mvtx)
            if hasattr(base_data[0], 'event_info'):
                return torch_geometric.data.Data(x_intt=x_intt, x_mvtx=x_mvtx, edge_index=edge_index, y=y, trigger=trigger, w=w, trigger_node=trigger_node, active_node=active_node, i=i, filename=filename, interaction_point=interaction_point, event_info=event_info, edge_index_intt=edge_index_intt, batch_intt=batch_intt, batch_mvtx=batch_mvtx, particle_id_intt=particle_id_intt, particle_id_mvtx=particle_id_mvtx, intt_tracks=intt_tracks, momentum_intt=momentum_intt)
            else:
                return torch_geometric.data.Data(x_intt=x_intt, x_mvtx=x_mvtx, edge_index=edge_index, y=y, trigger=trigger, w=w, trigger_node=trigger_node, active_node=active_node, i=i, filename=filename, interaction_point=interaction_point, edge_index_intt=edge_index_intt, batch_intt=batch_intt, batch_mvtx=batch_mvtx, particle_id_intt=particle_id_intt, particle_id_mvtx=particle_id_mvtx, intt_tracks=intt_tracks, momentum_intt=momentum_intt)


 
        collate_fn = collate_batch

    elif name == 'hit_graph_trigger_pileup':
        from torch_geometric.data import Batch
        from . import hit_graph_trigger_pileup
        train_dataset, valid_dataset = hit_graph_trigger_pileup.get_datasets(**data_args)
        collate_fn = Batch.from_data_list
    elif name == 'hetero_hitgraphs_sparse':
        from torch_geometric.data import Batch
        from . import hetero_hitgraphs_sparse
        train_dataset, valid_dataset = hetero_hitgraphs_sparse.get_datasets(**data_args)
        collate_fn = Batch.from_data_list
    else:
        raise Exception('Dataset %s unknown' % name)

    # Construct the data loaders
    loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
    train_sampler, valid_sampler = None, None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
        valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                   shuffle=(train_sampler is None), **loader_args)
    valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                         if valid_dataset is not None else None)
    return train_data_loader, valid_data_loader
