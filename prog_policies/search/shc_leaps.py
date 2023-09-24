from __future__ import annotations

import torch

from prog_policies.base import dsl_nodes

from .latent_cem_leaps import LatentCEM_LEAPS
from .utils import evaluate_program

class StochasticHillClimbing2_LEAPS(LatentCEM_LEAPS):
    
    def parse_method_args(self, search_method_args: dict):
        self.k = search_method_args.get('k', 250)
        self.sigma = search_method_args.get('sigma', 0.25)
        
    def init_search_vars(self):
        self.current_latent, self.current_program = self.random_latent()
        self.current_reward = evaluate_program(self.current_program, self.dsl, self.task_envs)
        self.num_evaluations = 1
        
    def get_search_vars(self) -> dict:
        return {
            'current_latent': self.current_latent,
            'current_program': self.current_program,
            'current_reward': self.current_reward,
        }
    
    def set_search_vars(self, search_vars: dict):
        self.current_latent = search_vars.get('current_latent')
        self.current_program = search_vars.get('current_program')
        self.current_reward = search_vars.get('current_reward')
    
    def decode_latent(self, latent: torch.Tensor) -> str:
        population = latent.unsqueeze(0)
        _, progs, progs_len, _, _, _, _, _, _ = self.latent_model.vae.decoder(None, population, teacher_enforcing=False, deterministic=True, evaluate=False)
        prog = progs.numpy().tolist()[0]
        prog_len = progs_len.numpy().tolist()[0][0]
        prog_str = self.leaps_dsl.intseq2str([0] + prog[:prog_len])
        prog = self.dsl.parse_str_to_node(prog_str)
        return prog
    
    def random_latent(self) -> torch.Tensor:
        """Samples a random latent vector from a normal distribution.

        Returns:
            torch.Tensor: Random latent vector.
        """
        while True:
            try:
                latent = torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
                prog = self.decode_latent(latent) # Check if it's a valid program
                break
            except (AssertionError, IndexError): # In case of invalid program, try again
                continue
        return latent, prog
    
    def search_iteration(self):
        if self.current_iteration % 100 == 0:
            self.log(f'Iteration {self.current_iteration}: Best reward {self.best_reward}, evaluations {self.num_evaluations}')
        
        if self.current_reward > self.best_reward:
            self.best_latent = self.current_latent
            self.best_reward = self.current_reward
            self.best_program = self.dsl.parse_node_to_str(self.current_program)
            self.save_best()
        if self.best_reward >= 1.0:
            return
        
        neighbors = []
        n_nodes = len(self.current_program.get_all_nodes())
        for _ in range(self.k):
            n_tries = 0
            while n_tries < 50:
                try:
                    if self.np_rng.rand() < 1/n_nodes:
                        neighbor = torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
                    else:
                        neighbor = self.current_latent + self.sigma * torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
                    prog = self.decode_latent(neighbor) # Check if it's a valid program
                    break
                except (AssertionError, IndexError): # In case of invalid program, try again
                    n_tries += 1
                    continue
            if n_tries == 50: raise Exception("Couldn't find a valid mutation")
            neighbors.append((neighbor, prog))
        
        in_local_maximum = True
        for latent, prog in neighbors:
            reward = evaluate_program(prog, self.dsl, self.task_envs)
            self.num_evaluations += 1
            if reward > self.current_reward:
                self.current_latent = latent
                self.current_program = prog
                self.current_reward = reward
                in_local_maximum = False
                break
        
        if in_local_maximum:
            self.current_latent, self.current_program = self.random_latent()
            self.current_reward = evaluate_program(self.current_program, self.dsl, self.task_envs)
            self.num_evaluations += 1
        