from __future__ import annotations

import torch

from prog_policies.base import dsl_nodes

from .latent_cem_leaps import LatentCEM_LEAPS
from .simulated_annealing import SimulatedAnnealing
from .utils import evaluate_program

class SimulatedAnnealing_LEAPS(LatentCEM_LEAPS, SimulatedAnnealing):
    
    def parse_method_args(self, search_method_args: dict):
        self.initial_temperature = search_method_args.get('initial_temperature', 100)
        self.initial_sigma = search_method_args.get('initial_sigma', 1.0)
        self.final_sigma = search_method_args.get('final_sigma', 0.05)
        self.sigma_decay = search_method_args.get('sigma_decay', 0.01)
        self.alpha = search_method_args.get('alpha', 0.9)
        self.beta = search_method_args.get('beta', 200)
        
    def init_search_vars(self):
        self.iterations_since_restart = 0
        self.current_latent = self.random_latent()
        self.current_program = self.decode_latent(self.current_latent)
        self.current_reward = evaluate_program(self.current_program, self.dsl, self.task_envs)
        self.current_temperature = self.initial_temperature
        self.current_sigma = self.initial_sigma
        self.best_latent = self.current_latent
        
    def get_search_vars(self) -> dict:
        return {
            'iterations_since_restart': self.iterations_since_restart,
            'current_latent': self.current_latent,
            'current_program': self.current_program,
            'current_reward': self.current_reward,
            'current_temperature': self.current_temperature,
            'current_sigma': self.current_sigma,
            'best_latent': self.best_latent
        }
    
    def set_search_vars(self, search_vars: dict):
        self.iterations_since_restart = search_vars.get('iterations_since_restart')
        self.current_latent = search_vars.get('current_latent')
        self.current_program = search_vars.get('current_program')
        self.current_reward = search_vars.get('current_reward')
        self.current_temperature = search_vars.get('current_temperature')
        self.current_sigma = search_vars.get('current_sigma')
        self.best_latent = search_vars.get('best_latent')
    
    def random_latent(self) -> torch.Tensor:
        """Samples a random latent vector from a normal distribution.

        Returns:
            torch.Tensor: Random latent vector.
        """
        return torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
    
    def decode_latent(self, latent: torch.Tensor) -> str:
        population = latent.unsqueeze(0)
        _, progs, progs_len, _, _, _, _, _, _ = self.latent_model.vae.decoder(None, population, teacher_enforcing=False, deterministic=True, evaluate=False)
        prog = progs.numpy().tolist()[0]
        prog_len = progs_len.numpy().tolist()[0][0]
        prog_str = self.leaps_dsl.intseq2str([0] + prog[:prog_len])
        return prog_str
    
    def search_iteration(self):
        if self.current_iteration % 100 == 0:
            self.log(f'Iteration {self.current_iteration}: Best reward {self.best_reward}, sigma {self.current_sigma}, evaluations {self.num_evaluations}')
        
        if self.current_reward > self.best_reward:
            self.best_latent = self.current_latent
            self.best_reward = self.current_reward
            self.best_program = self.current_program
            self.save_best()
        if self.best_reward >= 1.0:
            return
        
        if self.current_temperature > 1.0:
            next_latent = self.current_latent + self.current_sigma * torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
            next_program_str = self.decode_latent(next_latent)
            next_reward = evaluate_program(next_program_str, self.dsl, self.task_envs)
            self.num_evaluations += 1
            
            if next_reward > self.best_reward:
                self.best_latent = next_latent
                self.best_reward = next_reward
                self.best_program = next_program_str
                self.save_best()
            
            if self.np_rng.rand() < self.accept_function(self.current_reward, next_reward):
                self.current_latent = next_latent
                self.current_program = next_program_str
                self.current_reward = next_reward
                
            self.iterations_since_restart += 1
            self.decrease_temperature(self.iterations_since_restart)
            
        else:
            self.current_latent = self.random_latent()
            self.current_program = self.decode_latent(self.current_latent)
            self.current_reward = evaluate_program(self.current_program, self.dsl, self.task_envs)
            self.num_evaluations += 1

            self.current_temperature = self.initial_temperature
            self.iterations_since_restart = 0
            
            if self.current_sigma > self.final_sigma:
                self.current_sigma -= self.sigma_decay
        