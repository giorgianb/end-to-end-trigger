import wandb
from icecream import ic
import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)


import main_scripts.main_blri as blri
sweep_id ="jotosjgi"
entity = "giorgianborcatasciuc"
project = "physics-trigger"

GPU = 1
class ArgDict:
    pass

def train():
    wandb.init(tags=["biatt", "blri", "loopy", "layernorm", "batchnorm", "prelu"])
    config = {}
    for k, v in wandb.config.items():
        path = k.split('.')
        cur = config
        for item in path[:-1]:
            if item not in cur:
                cur[item] = {}
            cur = cur[item]
        cur[path[-1]] = v

    config['model']['layers_spec'] = [[64, 8], [64, 8], [64, 8], [64, 8]]
    config['wandb'] = {}
    config['wandb']['project_name'] = 'physics-trigger'
    config['wandb']['run_name'] = 'biatt-blri'
    config['wandb']['tags'] = ["biatt", "blri", "loopy", "layernorm", "prelu"]


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

    blri.execute_training(args, config)

wandb.agent(sweep_id=sweep_id, entity=entity, project=project, function=train)
