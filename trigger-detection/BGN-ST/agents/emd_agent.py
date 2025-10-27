import wandb
from icecream import ic
import sys
import os
import copy

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)


import main_scripts.main_emd as emd
sweep_id ="enq4rxgj"
entity = "giorgianborcatasciuc"
project = "physics-trigger"

class ArgDict:
    pass

model = None
def train():
    global model, train_data, val_data, test_data
    wandb.init(tags=["emd",  "loopy"])
    config = {}
    for k, v in wandb.config.items():
        path = k.split('.')
        cur = config
        for item in path[:-1]:
            if item not in cur:
                cur[item] = {}
            cur = cur[item]
        cur[path[-1]] = v

    config['wandb'] = {}
    config['wandb']['project_name'] = 'physics-trigger'
    config['wandb']['run_name'] = 'emd'
    config['wandb']['tags'] = ["emd", "loopy"]


    args = ArgDict()
    args.use_wandb = True
    args.save = True
    args.debug_load = False
    args.use_wandb = True
    args.skip_wandb_init = True
    args.verbose = False
    args.resume = False
    args.early_stopping = True
    args.early_stopping_accuracy = 0.6
    args.early_stopping_epoch = 1

    dconfig = copy.copy(config['data'])

    if model is None:
        train_data, val_data, test_data = emd.get_data_loaders(**dconfig)
        model = emd.fit_data(train_data, val_data, config['model'])

    emd.execute_evaluation(model, args, config)



wandb.agent(sweep_id=sweep_id, entity=entity, project=project, function=train)
