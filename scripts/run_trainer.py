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
    
    parser.add_argument('--name', default='LEAPS-PL-256', help='Name of the model')
    
    parser.add_argument('--model_args_path', default='sample_args/model/leaps_pl_256.json', help='Arguments path for VAE model')
    parser.add_argument('--model_seed', type=int, help='Seed for model initialization')
    
    parser.add_argument('--dataloader_args_path', default='sample_args/dataloader/leaps_dataset.json', help='Arguments path for program dataset')
    parser.add_argument('--dataloader_seed', type=int, help='Seed for data loading and shuffling')
    
    parser.add_argument('--disable_gpu', action='store_true', help='Disable GPU')
    
    parser.add_argument('--log_folder', default='logs', help='Folder to save logs')
    
    args = parser.parse_args()
    
    if args.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.model_args_path, 'r') as f:
        model_args = json.load(f)
    
    with open(args.dataloader_args_path, 'r') as f:
        dataloader_args = json.load(f)

    if args.model_seed is not None:
        model_args['params']['model_seed'] = args.model_seed

    if args.dataloader_seed is not None:
        dataloader_args['data_seed'] = args.dataloader_seed

    log_filename = f'{args.name}_{model_args["params"]["model_seed"]}'

    os.makedirs(args.log_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(args.log_folder, f'{log_filename}_{timestamp}.txt'), mode='w')
    ]
    
    logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.INFO)
    
    logger = logging.getLogger()
    dsl = KarelDSL()
    
    model = load_model(model_args["class"], {
        "dsl": dsl,
        "device": device,
        "logger": logger,
        "name": args.name,
        "env_cls": KarelEnvironment,
        "env_args": model_args["env_args"],
        **model_args["params"]
    })

    train_data, val_data, _ = make_dataloaders(dsl, device, **dataloader_args)
    
    model.fit(train_data, val_data, **model_args["trainer"])
    