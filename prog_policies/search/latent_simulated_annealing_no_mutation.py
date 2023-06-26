from __future__ import annotations
from functools import partial
import copy

import torch

from prog_policies.base import dsl_nodes

from .latent_simulated_annealing import LatentSimulatedAnnealing
from .utils import evaluate_program

class LatentSimulatedAnnealingNoMutation(LatentSimulatedAnnealing):

    def mutate_current_program(self) -> dsl_nodes.Program:
        current_program_tokens = self.dsl.parse_node_to_int(self.current_program)
        current_program_tensor = torch.tensor(current_program_tokens, dtype=torch.long,
                                              device=self.torch_device)
        starting_latent = self.latent_model.encode_program(current_program_tensor).detach()
        starting_latent = starting_latent.repeat(self.cem_pop_size, 1)
        best_program_so_far = self.current_program
        best_reward_so_far = self.current_reward
        
        # CEM to find a program
        population = starting_latent + self.cem_sigma * torch.randn(self.cem_pop_size, self.latent_model.hidden_size,
                                                                    generator=self.torch_rng, device=self.torch_device)
        
        for cem_iter in range(1, self.cem_iterations + 1):
            population_tokens = self.latent_model.decode_vector(population)
            population_str = [self.dsl.parse_int_to_str(prog_tokens) for prog_tokens in population_tokens]
            
            if self.pool is not None:
                fn = partial(evaluate_program, dsl=self.dsl, task_envs=self.task_envs)
                rewards = self.pool.map(fn, population_str)
            else:
                rewards = [evaluate_program(p, self.dsl, self.task_envs) for p in population_str]
            
            for r, prog_str in zip(rewards, population_str):
                self.num_evaluations += 1
                if r > best_reward_so_far:
                    best_program_so_far = self.dsl.parse_str_to_node(prog_str)
                    best_reward_so_far = r
                if r > self.best_reward:
                    self.best_reward = r
                    self.best_program = prog_str
                    self.save_best()
                    
                if self.best_reward >= 1.0:
                    return self.dsl.parse_str_to_node(prog_str)
                
            n_elite = int(self.cem_elitism * self.cem_pop_size)
            best_indices = torch.topk(torch.tensor(rewards, device=self.torch_device), n_elite).indices
            elite_population = population[best_indices]
            
            new_indices = torch.ones(elite_population.size(0), device=self.torch_device).multinomial(
                self.cem_pop_size, generator=self.torch_rng, replacement=True)
            new_population = []
            for index in new_indices:
                sample = elite_population[index]
                new_population.append(
                    sample + self.cem_sigma * torch.randn(self.latent_model.hidden_size,
                                                          generator=self.torch_rng,
                                                          device=self.torch_device)
                )
            population = torch.stack(new_population)
        
        return best_program_so_far
