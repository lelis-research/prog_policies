from __future__ import annotations
from functools import partial

import torch

from .base_search import BaseSearch
from .utils import evaluate_program

class LatentCEM(BaseSearch):
    """Implements the CEM method from LEAPS paper.
    """
    def parse_method_args(self, search_method_args: dict):
        self.population_size = search_method_args.get('population_size', 32)
        self.elitism_rate = search_method_args.get('elitism_rate', 0.1)
        self.restart_timeout = search_method_args.get('restart_timeout', 10)
        self.initial_sigma = search_method_args.get('initial_sigma', 0.1)
        self.reduce_to_mean = search_method_args.get('reduce_to_mean', False)
        self.exp_decay = search_method_args.get('exp_decay', False)
    
    def init_search_vars(self):
        self.sigma = self.initial_sigma
        self.counter_for_restart = 0
        self.best_mean_elite_reward_since_restart = -float('inf')
        self.population = self.init_population()
        self.converged = False
        
    def get_search_vars(self) -> dict:
        return {
            'sigma': self.sigma,
            'counter_for_restart': self.counter_for_restart,
            'best_mean_elite_reward_since_restart': self.best_mean_elite_reward_since_restart,
            'population': self.population,
            'converged': self.converged
        }
        
    def set_search_vars(self, search_vars: dict):
        self.sigma = search_vars.get('sigma')
        self.counter_for_restart = search_vars.get('counter_for_restart')
        self.best_mean_elite_reward_since_restart = search_vars.get('best_mean_elite_reward_since_restart')
        self.population = search_vars.get('population')
        self.converged = search_vars.get('converged')
    
    def init_population(self) -> torch.Tensor:
        """Initializes the CEM population from a normal distribution.

        Returns:
            torch.Tensor: Initial population as a tensor.
        """
        return torch.randn(self.population_size, self.hidden_size,
                           generator=self.torch_rng, device=self.torch_device)
        
    def decode_population(self, population: torch.Tensor) -> list[str]:
        programs_tokens = self.latent_model.decode_vector(population)
        return [self.dsl.parse_int_to_str(prog_tokens) for prog_tokens in programs_tokens]
    
    def execute_population(self, population: torch.Tensor) -> tuple[list[str], torch.Tensor, int]:
        programs_str = self.decode_population(population)
        
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
        
        return torch.tensor(rewards, device=self.torch_device)

    def search_iteration(self):
        rewards = self.execute_population(self.population)
        
        if self.converged:
            return
        
        n_elite = int(self.population_size * self.elitism_rate)
        best_indices = torch.topk(rewards, n_elite).indices
        elite_population = self.population[best_indices]
        mean_elite_reward = torch.mean(rewards[best_indices])
        std_elite_reward = torch.std(rewards[best_indices])

        self.log(f'Iteration {self.current_iteration} elite reward mean: {mean_elite_reward}, std: {std_elite_reward}')
        
        if mean_elite_reward.cpu().numpy() <= self.best_mean_elite_reward_since_restart:
            self.counter_for_restart += 1
        else:
            self.counter_for_restart = 0
            self.best_mean_elite_reward_since_restart = mean_elite_reward.cpu().numpy()
        
        if self.counter_for_restart >= self.restart_timeout and self.restart_timeout > 0:
            self.init_search_vars()
            self.log('Restarted population.')
        else:
            new_indices = torch.ones(elite_population.size(0), device=self.torch_device).multinomial(
                self.population_size, generator=self.torch_rng, replacement=True)
            if self.reduce_to_mean:
                elite_population = torch.mean(elite_population, dim=0).repeat(n_elite, 1)
            new_population = []
            for index in new_indices:
                sample = elite_population[index]
                new_population.append(
                    sample + self.sigma * torch.randn(self.hidden_size,
                                                      generator=self.torch_rng,
                                                      device=self.torch_device)
                )
            self.population = torch.stack(new_population)

        if self.exp_decay:
            self.sigma *= 0.998
            if self.sigma < 0.1:
                self.sigma = 0.1