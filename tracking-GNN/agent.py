import wandb
from icecream import ic
import sys
import os
import traceback

import train as main
sweep_id = "f4y16pbd"
entity = "giorgianborcatasciuc"
project = "particle-model-tree"

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)




GPU = 0

class ArgDict:
    pass


def train():
    print(f'GPU: {GPU}')
    wandb.init()
    config = {}
    for k, v in wandb.config.items():
        path = k.split('.')
        cur = config
        for item in path[:-1]:
            if item not in cur:
                cur[item] = {}
            cur = cur[item]
        cur[path[-1]] = v

    config['wandb'] = 'agnn-trigger'
    config['model_name_on_wandb'] = 'agnn'
    config['lr_decay_schedule'] = [{'start_epoch': 0, 'end_epoch': 10, 'factor': config['optimizer']['learning_rate_decay_rate']}]
    del config['optimizer']['learning_rate_decay_rate']

    args = ArgDict()
    args.rank_gpu = None
    args.verbose = None
    args.ranks_per_node = 8
    args.resume = None
    args.interactive = None
    args.output_dir = None
    args.fom = None
    args.n_train = None
    args.n_valid = None
    args.batch_size = None
    args.n_epochs = None
    args.real_weight = None
    args.lr= None
    args.hidden_dim = None
    args.n_graph_iters = None
    args.weight_decay = None
    args.wandb = 'physics-trigger'
    args.use_wandb = True

    args.show_config = False
    args.seed = 42
    args.ranks_per_node = 8 
    args.use_wandb = True
    args.gpu = GPU
    args.eva = 1
    args.save = True
    args.skip_wandb_init = True
    args.verbose = False
    args.resume = False
    args.distributed = None

    try:
        main.execute_training(args, config)
    except Exception:
        print(traceback.format_exc())
        raise


wandb.agent(sweep_id=sweep_id, entity=entity, project=project, function=train)
