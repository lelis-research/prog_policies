from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from glob import glob
import json
import logging
import os
import pickle
from logging import Logger
from multiprocessing import Pool
import sys

import numpy as np
import torch
import wandb

from prog_policies.base import BaseDSL, BaseEnvironment
from prog_policies.latent_space.models import load_model
from prog_policies.karel_tasks import get_task_cls

class BaseSearch(ABC):
    
    def __init__(self, dsl: BaseDSL, env_cls: type[BaseEnvironment],
                 device: torch.device, task_cls_name: str, 
                 env_args: dict, number_executions: int, exp_name: str,
                 search_method_args: dict, max_evaluations: int,
                 latent_model_cls_name: str = None, latent_model_args: dict = {},
                 latent_model_params_path: str = None, search_seed: int = 1,
                 checkpoint_frequency: int = 100, base_output_folder: str = 'output',
                 base_checkpoint_folder: str = 'checkpoints', log_folder: str = None,
                 n_proc: int = 1,
                 method_label: str = None, wandb_args: dict = None, use_checkpoint: bool = True):
        self.dsl = dsl
        task_cls = get_task_cls(task_cls_name)
        self.task_envs = [task_cls(env_args, i) for i in range(number_executions)]
        env_width = self.task_envs[0].initial_environment.state_shape[2]
        env_height = self.task_envs[0].initial_environment.state_shape[1]
        if self.task_envs[0].initial_environment.crashable:
            env_behaviour = 'Crashable'
        elif self.task_envs[0].initial_environment.leaps_behaviour:
            env_behaviour = 'LeapsBehaviour'
        else:
            env_behaviour = 'Standard'
        task_specifier = f'{task_cls_name}_{env_width}x{env_height}_{env_behaviour}'
        if method_label is None:
            method_label = self.__class__.__name__
        self.output_folder = os.path.join(base_output_folder, exp_name, 'search', method_label)
        os.makedirs(self.output_folder, exist_ok=True)
        self.parse_method_args(search_method_args)
        with open(os.path.join(self.output_folder, 'search_args.json'), 'w') as f:
            json.dump(search_method_args, f)
        self.task_output_folder = os.path.join(self.output_folder, task_specifier)
        self.checkpoint_folder = os.path.join(base_checkpoint_folder, exp_name, 'search',
                                              method_label, task_specifier)
        os.makedirs(self.task_output_folder, exist_ok=True)
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        self.max_evaluations = max_evaluations
        self.n_proc = n_proc
        self.torch_device = device
        if latent_model_cls_name is not None:
            self.load_latent_model(latent_model_cls_name, latent_model_args,
                                   latent_model_params_path, env_cls, env_args)
        self.search_seed = search_seed
        self.checkpoint_frequency = checkpoint_frequency
        if log_folder is not None:
            log_filename = f'{method_label}_{task_specifier}_{search_seed}'
            os.makedirs(log_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_handlers = [
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(log_folder, f'{log_filename}_{timestamp}.txt'), mode='w')
            ]
            logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.INFO)
            self.logger = logging.getLogger()
        else:
            self.logger = None
        if wandb_args:
            group_name = f'{method_label}_{task_specifier}'
            run_name = f'{group_name}_{search_seed}'
            self.wandb_run = wandb.init(
                group=group_name,
                name=run_name,
                config={
                    'search_seed': search_seed,
                    'task_cls_name': task_cls_name,
                    'env_args': env_args,
                    'number_executions': number_executions,
                    'exp_name': exp_name,
                    'search_method_args': search_method_args,
                    'max_evaluations': max_evaluations,
                    'latent_model_cls_name': latent_model_cls_name,
                    'latent_model_args': latent_model_args,
                    'latent_model_params_path': latent_model_params_path,
                    'checkpoint_frequency': checkpoint_frequency,
                    'method_label': method_label
                },
                **wandb_args
            )
        else:
            self.wandb_run = None
        self.use_checkpoint = use_checkpoint
    
    @abstractmethod
    def parse_method_args(self, search_method_args: dict):
        pass
    
    def load_latent_model(self, latent_model_cls_name: str, latent_model_args: dict,
                          latent_model_params_path: str, env_cls: type[BaseEnvironment] = None,
                          env_args: dict = None):
        latent_model_args['device'] = self.torch_device
        latent_model_args['dsl'] = self.dsl
        latent_model_args['env_args'] = env_args
        latent_model_args['env_cls'] = env_cls
        torch.set_num_threads(self.n_proc)
        self.latent_model = load_model(latent_model_cls_name, latent_model_args,
                                        latent_model_params_path)
        self.hidden_size = self.latent_model.hidden_size
    
    def init_search_state(self):
        self.np_rng = np.random.RandomState(self.search_seed)
        self.torch_rng = torch.Generator(device=self.torch_device)
        self.torch_rng.manual_seed(self.np_rng.randint(1000000))
        self.current_iteration = 1
        self.best_program = ''
        self.best_reward = -np.inf
        self.num_evaluations = 0
        self.init_search_vars()
        
    def init_search_vars(self):
        pass
    
    def get_search_state(self) -> dict:
        return {
            'np_rng_state': self.np_rng.get_state(),
            'torch_rng_state': self.torch_rng.get_state(),
            'current_iteration': self.current_iteration,
            'best_program': self.best_program,
            'best_reward': self.best_reward,
            'num_evaluations': self.num_evaluations,
            'search_vars': self.get_search_vars()
        }
    
    def get_search_vars(self) -> dict:
        return {}
    
    def set_search_state(self, search_state: dict):
        self.np_rng.set_state(search_state.get('np_rng_state'))
        self.torch_rng.set_state(search_state.get('torch_rng_state'))
        self.current_iteration = search_state.get('current_iteration')
        self.best_program = search_state.get('best_program')
        self.best_reward = search_state.get('best_reward')
        self.num_evaluations = search_state.get('num_evaluations')
        self.set_search_vars(search_state.get('search_vars'))
        
    def set_search_vars(self, search_vars: dict):
        pass
    
    @abstractmethod
    def search_iteration(self):
        pass
    
    def search(self) -> tuple[str, float, int]:
        self.init_search_state()
        checkpoints = glob(os.path.join(self.checkpoint_folder, f'seed_{self.search_seed}_iter_*.pkl'))
        # Load checkpoint if exists
        if len(checkpoints) > 0 and self.use_checkpoint:
            checkpoints.sort(key=os.path.getmtime)
            checkpoint = checkpoints[-1]
            with open(checkpoint, 'rb') as f:
                search_state = pickle.load(f)
                self.set_search_state(search_state)
            self.log(f'Loaded checkpoint {checkpoint}')
            self.current_iteration += 1
        else:
            self.init_output()
        
        if self.n_proc > 1:
            self.pool = Pool(self.n_proc)
        else:
            self.pool = None
        
        # Main search loop, assumes search can be separated into iterations following search_iteration
        while self.num_evaluations < self.max_evaluations:
            self.search_iteration()
            if self.current_iteration % self.checkpoint_frequency == 0:
                existing_checkpoints = glob(os.path.join(self.checkpoint_folder,
                                                         f'seed_{self.search_seed}_iter_*.pkl'))
                # Save current checkpoint
                checkpoint_fname = f'seed_{self.search_seed}_iter_{self.current_iteration}.pkl'
                with open(os.path.join(self.checkpoint_folder, checkpoint_fname), 'wb') as f:
                    pickle.dump(self.get_search_state(), f)
                # Remove old checkpoints
                for checkpoint in existing_checkpoints:
                    os.remove(checkpoint)
            if self.best_reward >= 1.:
                break
            self.current_iteration += 1
        
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
        
        # When search finishes, remove all checkpoints
        all_checkpoints = glob(os.path.join(self.checkpoint_folder, f'seed_{self.search_seed}_iter_*.pkl'))
        for checkpoint in all_checkpoints:
            os.remove(checkpoint)

        # Touch a file to indicate search is finished
        open(os.path.join(self.checkpoint_folder, f'seed_{self.search_seed}_finished'), 'a').close()
        
        # Save last seen program if not converged
        if self.best_reward < 1.:
            self.log('Search finished without finding a solution')
            self.save_best()
        
        return self.best_program, self.best_reward, self.num_evaluations
    
    def log(self, message: str):
        if self.logger is not None:
            self.logger.info(f'[{self.__class__.__name__}] {message}')
        else:
            print(f'[{self.__class__.__name__}] {message}')
    
    def init_output(self):
        fields = ['num_evaluations', 'best_reward', 'best_program']
        with open(os.path.join(self.task_output_folder, f'seed_{self.search_seed}.csv'), 'w') as f:
            f.write(','.join(fields))
            f.write('\n')

    def _log_wandb(self):
        self.wandb_run.log(dict(
                num_evaluations=self.num_evaluations,
                best_program=self.best_program,
                best_reward=self.best_reward,
                **{k: v for k, v in self.get_search_vars().items() if isinstance(v, (int, float))}
            ), step=self.num_evaluations)
    
    def save_best(self):
        fields = [self.num_evaluations, self.best_reward, self.best_program]
        with open(os.path.join(self.task_output_folder, f'seed_{self.search_seed}.csv'), 'a') as f:
            f.write(','.join([str(i) for i in fields]))
            f.write('\n')
        self.log(f'New best program: {self.best_program}')
        self.log(f'New best reward: {self.best_reward}')
        self.log(f'Number of evaluations: {self.num_evaluations}')
        if self.wandb_run:
            self._log_wandb()
