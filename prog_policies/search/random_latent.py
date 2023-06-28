from __future__ import annotations
from functools import partial

import torch

from .base_search import BaseSearch
from .utils import evaluate_program

class RandomLatent(BaseSearch):
    def parse_method_args(self, search_method_args: dict):
        self.population_size = search_method_args.get('population_size', 32)
        self.initial_sigma = search_method_args.get('initial_sigma', 0.1)
    
    def init_search_vars(self):
        self.sigma = self.initial_sigma
        self.population = self.init_population()
        self.converged = False
        
    def get_search_vars(self) -> dict:
        return {
            'sigma': self.sigma,
            'population': self.population,
            'converged': self.converged
        }
        
    def set_search_vars(self, search_vars: dict):
        self.sigma = search_vars.get('sigma')
        self.population = search_vars.get('population')
        self.converged = search_vars.get('converged')
    
    def init_population(self) -> torch.Tensor:
        """Initializes the CEM population from a normal distribution.

        Returns:
            torch.Tensor: Initial population as a tensor.
        """
        return torch.randn(self.population_size, self.latent_model.hidden_size,
                           generator=self.torch_rng, device=self.torch_device)
        
        
    def execute_population(self, population: torch.Tensor) -> tuple[list[str], torch.Tensor, int]:
        programs_tokens = self.latent_model.decode_vector(population)
        programs_str = [self.dsl.parse_int_to_str(prog_tokens) for prog_tokens in programs_tokens]
        
        if self.pool is not None:
            fn = partial(evaluate_program, dsl=self.dsl, task_envs=self.task_envs)
            rewards = self.pool.map(fn, programs_str)
        else:
            rewards = [evaluate_program(p, self.dsl, self.task_envs) for p in programs_str]
        
        for r, prog_str in zip(rewards, programs_str):
            self.num_evaluations += 1
            if r > self.best_reward:
                self.best_reward = r
                self.best_program = prog_str
                self.save_best()
                
            if self.best_reward >= 1.0:
                self.converged = True
                break

    def search_iteration(self):
        self.execute_population(self.population)
        self.log(f'Iteration {self.current_iteration}: Best reward {self.best_reward}, evaluations {self.num_evaluations}')
        
        if self.converged:
            return
        
        self.population += self.sigma * torch.randn(self.population_size, self.latent_model.hidden_size,
                                                     generator=self.torch_rng, device=self.torch_device)
