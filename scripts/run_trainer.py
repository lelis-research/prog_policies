import datetime
import json
import logging
import os
import torch
import sys

sys.path.append('.')

from prog_policies.karel import KarelDSL, KarelEnvironment
from prog_policies.latent_space.models import load_model
from prog_policies.latent_space.program_dataset import make_dataloaders
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

if __name__ == '__main__':
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--config_path', default='sample_args/trainer/leaps_pl_256.json', help='Path for trainer config')

    parser.add_argument('--disable_gpu', action='store_true', help='Disable GPU')
    
    parser.add_argument('--log_folder', default='logs', help='Folder to save logs')
    
    parser.add_argument('--wandb_entity', type=str, help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, help='Wandb project')
    
    args = parser.parse_args()
    
    if args.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    log_filename = f'{config["label"]}_{config["model"]["params"]["model_seed"]}'

    os.makedirs(args.log_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(args.log_folder, f'{log_filename}_{timestamp}.txt'), mode='w')
    ]
    
    logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.INFO)
    
    logger = logging.getLogger()
    
    if args.wandb_project:
        wandb_args = {
            'project': args.wandb_project,
            'entity': args.wandb_entity,
            'config': config
        }
    else:
        wandb_args = None
    
    dsl = KarelDSL()
    
    model = load_model(config["model"]["class"], {
        "dsl": dsl,
        "device": device,
        "logger": logger,
        "name": config["label"],
        "env_cls": KarelEnvironment,
        "env_args": config["model"]["env_args"],
        "wandb_args": wandb_args,
        **config["model"]["params"]
    })

    train_data, val_data, _ = make_dataloaders(dsl, device, **config["dataloader"])
    
    model.fit(train_data, val_data, **config["model"]["trainer"])
    