from __future__ import annotations
from functools import partial

import torch

from .base_search import BaseSearch
from .utils import evaluate_program

class DisentangledLatentCEM(BaseSearch):
    """Implements the CEM method from LEAPS paper.
    """
    def parse_method_args(self, search_method_args: dict):
        self.population_size = search_method_args.get('population_size', 32)
        self.elitism_rate = search_method_args.get('elitism_rate', 0.1)
        self.restart_timeout = search_method_args.get('restart_timeout', 10)
        self.initial_sigma = search_method_args.get('initial_sigma', 0.1)
        self.reduce_to_mean = search_method_args.get('reduce_to_mean', False)
    
    def init_search_vars(self):
        self.sigma = self.initial_sigma
        self.sem_counter_for_restart = 0
        self.syn_counter_for_restart = 0
        self.sem_population = self.init_sem_population()
        self.syn_population = self.init_syn_population()
        self.sem_best_mean_elite_reward_since_restart = -float('inf')
        self.syn_best_mean_elite_reward_since_restart = -float('inf')
        self.converged = False
        
    def get_search_vars(self) -> dict:
        return {
            'sigma': self.sigma,
            'sem_counter_for_restart': self.sem_counter_for_restart,
            'syn_counter_for_restart': self.syn_counter_for_restart,
            'sem_best_mean_elite_reward_since_restart': self.sem_best_mean_elite_reward_since_restart,
            'syn_best_mean_elite_reward_since_restart': self.syn_best_mean_elite_reward_since_restart,
            'sem_population': self.sem_population,
            'syn_population': self.syn_population,
            'converged': self.converged
        }
        
    def set_search_vars(self, search_vars: dict):
        self.sigma = search_vars.get('sigma')
        self.sem_counter_for_restart = search_vars.get('sem_counter_for_restart')
        self.syn_counter_for_restart = search_vars.get('syn_counter_for_restart')
        self.sem_best_mean_elite_reward_since_restart = search_vars.get('sem_best_mean_elite_reward_since_restart')
        self.syn_best_mean_elite_reward_since_restart = search_vars.get('syn_best_mean_elite_reward_since_restart')
        self.sem_population = search_vars.get('sem_population')
        self.syn_population = search_vars.get('syn_population')
        self.converged = search_vars.get('converged')
    
    def init_sem_population(self) -> torch.Tensor:
        return torch.randn(self.population_size, self.latent_model.sem_latent_size,
                           generator=self.torch_rng, device=self.torch_device)
        
    def init_syn_population(self) -> torch.Tensor:
        return torch.randn(self.population_size, self.latent_model.syn_latent_size,
                           generator=self.torch_rng, device=self.torch_device)
        
    def execute_programs(self, programs_tokens) -> torch.Tensor:
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
        
        return torch.tensor(rewards, device=self.torch_device)

    def search_iteration(self):
        
        # latents is the cross product of every syn and sem latent
        latents = []
        for sem_latent in self.sem_population:
            for syn_latent in self.syn_population:
                latents.append(torch.stack([sem_latent, syn_latent]))
        
        latents = torch.stack(latents)
        
        programs_tokens = self.latent_model.decode_vector(latents[:, 1], latents[:, 0])
        
        rewards = self.execute_programs(programs_tokens)
        
        if self.converged:
            return
        
        # each row is a sem latent, each column is a syn latent
        rewards = rewards.reshape(self.population_size, self.population_size)
        
        sem_score = torch.max(rewards, dim=1).values
        syn_score = torch.max(rewards, dim=0).values
        
        n_elite = int(self.population_size * self.elitism_rate)

        sem_best_indices = torch.topk(sem_score, n_elite).indices
        sem_elite_population = self.sem_population[sem_best_indices]
        sem_mean_elite_reward = torch.mean(sem_score[sem_best_indices])
        
        syn_best_indices = torch.topk(syn_score, n_elite).indices
        syn_elite_population = self.syn_population[syn_best_indices]
        syn_mean_elite_reward = torch.mean(syn_score[syn_best_indices])

        self.log(f'Iteration {self.current_iteration} sem elite mean: {sem_mean_elite_reward}, syn elite mean: {syn_mean_elite_reward}')

        if sem_mean_elite_reward.cpu().numpy() == self.sem_best_mean_elite_reward_since_restart:
            self.sem_counter_for_restart += 1
        else:
            self.sem_counter_for_restart = 0
            self.sem_best_mean_elite_reward_since_restart = sem_mean_elite_reward.cpu().numpy()
            
        if syn_mean_elite_reward.cpu().numpy() == self.syn_best_mean_elite_reward_since_restart:
            self.syn_counter_for_restart += 1
        else:
            self.syn_counter_for_restart = 0
            self.syn_best_mean_elite_reward_since_restart = syn_mean_elite_reward.cpu().numpy()
            
        if (self.sem_counter_for_restart >= self.restart_timeout or self.syn_counter_for_restart >= self.restart_timeout) and self.restart_timeout > 0:
            self.sem_population = self.init_sem_population()
            self.sem_best_mean_elite_reward_since_restart = -float('inf')
            self.sem_counter_for_restart = 0
            self.syn_population = self.init_syn_population()
            self.syn_best_mean_elite_reward_since_restart = -float('inf')
            self.syn_counter_for_restart = 0
            self.log('Restarted both populations.')
        else:
            sem_new_indices = torch.ones(sem_elite_population.size(0), device=self.torch_device).multinomial(
                self.population_size, generator=self.torch_rng, replacement=True)
            if self.reduce_to_mean:
                sem_elite_population = torch.mean(sem_elite_population, dim=0).repeat(n_elite, 1)
                self.sigma = torch.std(sem_elite_population)
            sem_new_population = []
            for index in sem_new_indices:
                sample = sem_elite_population[index]
                sem_new_population.append(
                    sample + self.sigma * torch.randn(self.latent_model.sem_latent_size,
                                                      generator=self.torch_rng,
                                                      device=self.torch_device)
                )
            self.sem_population = torch.stack(sem_new_population)
            
            syn_new_indices = torch.ones(syn_elite_population.size(0), device=self.torch_device).multinomial(
                self.population_size, generator=self.torch_rng, replacement=True)
            if self.reduce_to_mean:
                syn_elite_population = torch.mean(syn_elite_population, dim=0).repeat(n_elite, 1)
                self.sigma = torch.std(syn_elite_population)
            syn_new_population = []
            for index in syn_new_indices:
                sample = syn_elite_population[index]
                syn_new_population.append(
                    sample + self.sigma * torch.randn(self.latent_model.syn_latent_size,
                                                      generator=self.torch_rng,
                                                      device=self.torch_device)
                )
            self.syn_population = torch.stack(syn_new_population)
