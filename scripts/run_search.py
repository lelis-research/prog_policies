from datetime import datetime
import json
import logging
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import wandb

sys.path.append('.')

from prog_policies.karel import KarelDSL, KarelEnvironment
from prog_policies.latent_space.models import *
from prog_policies.search import get_search_cls

if __name__ == '__main__':
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--search_method', default='LatentCEM', help='Name of search method class')
    parser.add_argument('--search_args_path', default='sample_args/search/latent_cem.json', help='Arguments path for search method')
    parser.add_argument('--log_folder', default='logs', help='Folder to save logs')
    parser.add_argument('--search_seed', type=int, help='Seed for search method')
    parser.add_argument('--wandb_entity', type=str, help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, help='Wandb project')
    
    args = parser.parse_args()
    
    with open(args.search_args_path, 'r') as f:
        search_args = json.load(f)
    
    if args.search_seed is not None:
        search_args['search_seed'] = args.search_seed
    
    log_filename = f'{args.search_method}_{search_args["task_cls_name"]}_{search_args["search_seed"]}'

    os.makedirs(args.log_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        }
    else:
        wandb_args = None
    
    device = torch.device('cpu')
    
    dsl = KarelDSL()
    
    search_cls = get_search_cls(args.search_method)
    
    searcher = search_cls(dsl, KarelEnvironment, device, logger=logger, wandb_args=wandb_args, **search_args)
    
    searcher.search()
