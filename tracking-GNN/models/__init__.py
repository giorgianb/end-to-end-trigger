"""
Python module for holding our PyTorch models.
"""

def get_model(name, **model_args):
    """
    Top-level factory function for getting your models.
    """
    if name == 'agnn_original':
        from .agnn_original import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'agnn':
        from .agnn import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'mpnn':
        from .mpnn import GNN
        return GNN(**model_args)
    elif name == 'noise_agnn':
        from .noise_agnn import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'agnn_trigger':
        from .agnn_trigger import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)
    elif name == 'agat_trigger':
        from .agat_trigger import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)
    elif name == 'agat_trigger_depileup':
        from .agat_trigger_depileup import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)
    elif name == 'agat_trigger_node_depileup':
        from .agat_trigger_node_depileup import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)
    elif name == 'agat_active':
        from .agat_active import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)
    elif name == 'garnet_trigger':
        from .garnet_trigger import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)
    elif name == 'garnet_active':
        from .garnet_active import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)
    elif name == 'mlp_trigger':
        from .mlp_trigger import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)
    elif name == 'mlp_trigger_layerwise':
        from .mlp_trigger_layerwise import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)
    elif name == 'hetero_gnn':
        from .heterogeneous_gnn import HeteroGNNSegmentClassifier
        return HeteroGNNSegmentClassifier(**model_args)
    elif name == 'garnet_tracking':
        from .garnet_tracking import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'bgn_st_tracking':
        from .bgn_st_tracking import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'bgn_st_track':
        from .bgn_st_track import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'bgn_st_trigger':
        from .bgn_st_trigger import GNNGraphClassifier
        return GNNGraphClassifier(**model_args)

    else:
        raise Exception('Model %s unknown' % name)
