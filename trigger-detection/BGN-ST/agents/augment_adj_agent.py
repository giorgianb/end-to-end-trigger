import wandb
from icecream import ic
import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)


import main_scripts.main_augment_track_embeddings as main
sweep_id = "no7qxngx"
entity = "giorgianborcatasciuc"
project = "physics-trigger"

GPU = 0
class ArgDict:
    pass

def train():
    print(f'GPU: {GPU}')
    wandb.init(tags=["biatt", "glri", "loopy", "layernorm", "batchnorm", "prelu", "4-layer"])
    config = {}
    for k, v in wandb.config.items():
        path = k.split('.')
        cur = config
        for item in path[:-1]:
            if item not in cur:
                cur[item] = {}
            cur = cur[item]
        cur[path[-1]] = v

    config['model']['layers_spec'] = [[64, 8], [32, 8], [32, 8], [32, 8]]
    config['wandb'] = {}
    config['wandb']['project_name'] = 'physics-trigger'
    config['wandb']['run_name'] = 'biatt-pileup-20'
    config['wandb']['tags'] = ["biatt", "glri", "loopy", "layernorm", "prelu", "4-layer"]


    args = ArgDict()
    args.use_wandb = True
    args.gpu = GPU
    args.save = True
    args.debug_load = False
    args.use_wandb = True
    args.skip_wandb_init = True
    args.verbose = False
    args.resume = False
    args.early_stopping = True
    args.early_stopping_accuracy = 0.6
    args.early_stopping_epoch = 1

    main.execute_training(args, config)

wandb.agent(sweep_id=sweep_id, entity=entity, project=project, function=train)
