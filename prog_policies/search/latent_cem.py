from __future__ import annotations
from functools import partial
from multiprocessing import Pool
import time
import torch

from prog_policies.base import BaseDSL, BaseTask, dsl_nodes
from prog_policies.latent_space.models import BaseVAE
from prog_policies.output_handler import OutputHandler

from .utils import evaluate_program

class LatentCEM:
    """Implements the CEM method from LEAPS paper.
    """
    def __init__(self, model: BaseVAE, task_cls: type[BaseTask], dsl: BaseDSL,
                 env_args: dict, population_size: int = 32, elitism_rate: float = 0.1,
                 number_executions: int = 16, number_iterations: int = 1000,
                 restart_timeout: int = 10, sigma: float = 0.1, reduce_to_mean: bool = False,
                 output_handler: OutputHandler = None, multiprocessing: bool = False):
        self.model = model
        self.dsl = dsl
        self.device = self.model.device
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.n_elite = int(elitism_rate * self.population_size)
        self.number_executions = number_executions
        self.number_iterations = number_iterations
        self.reduce_to_mean = reduce_to_mean
        self.sigma = sigma
        self.task_envs = [task_cls(env_args, i) for i in range(self.number_executions)]
        self.restart_timeout = restart_timeout
        self.output_handler = output_handler
        self.multiprocessing = multiprocessing
        if self.output_handler is not None:
            self.output_handler.setup_search()

    def _log(self, message: str):
        if self.output_handler is not None:
            self.output_handler.log('Latent CEM', message)

    def _save_best(self):
        if self.output_handler is not None:
            t = time.time() - self.start_time
            self.output_handler.save_search_info(t, self.num_evaluations, self.best_reward,
                                                    self.best_program)

    def init_population(self) -> torch.Tensor:
        """Initializes the CEM population from a normal distribution.

        Returns:
            torch.Tensor: Initial population as a tensor.
        """
        return torch.stack([
            torch.randn(self.model.hidden_size, device=self.device) for _ in range(self.population_size)
        ])
        
        
    def execute_population(self, population: torch.Tensor) -> tuple[list[str], torch.Tensor, int]:
        programs_tokens = self.model.decode_vector(population)
        programs_str = [self.dsl.parse_int_to_str(prog_tokens) for prog_tokens in programs_tokens]
        
        if self.multiprocessing:
            with Pool() as pool:
                fn = partial(evaluate_program, dsl=self.dsl, task_envs=self.task_envs)
                rewards = pool.map(fn, programs_str)
        else:
            rewards = [evaluate_program(p, self.dsl, self.task_envs) for p in programs_str]
        
        for r, prog_str in zip(rewards, programs_str):
            self.num_evaluations += 1
            if r > self.best_reward:
                self.best_reward = r
                self.best_program = prog_str
                self._log(f'New best reward: {self.best_reward}')
                self._log(f'New best program: {self.best_program}')
                self._log(f'Number of evaluations: {self.num_evaluations}')
                self._save_best()
                
            if self.best_reward >= 1.0:
                self.converged = True
                break                
        
        return torch.tensor(rewards, device=self.device)

    
    def search(self) -> tuple[str, bool, int]:
        population = self.init_population()
        self.converged = False
        self.num_evaluations = 0
        counter_for_restart = 0
        self.best_reward = -float('inf')
        self.best_program = None
        prev_mean_elite_reward = -float('inf')
        self.start_time = time.time()
        if self.output_handler is not None:
            self.output_handler.setup_search_info('latent_cem')

        for iteration in range(1, self.number_iterations + 1):
            rewards = self.execute_population(population)
            
            if self.converged:
                break
            
            best_indices = torch.topk(rewards, self.n_elite).indices
            elite_population = population[best_indices]
            mean_elite_reward = torch.mean(rewards[best_indices])

            self._log(f'Iteration {iteration} mean elite reward: {mean_elite_reward}')
            
            if mean_elite_reward.cpu().numpy() == prev_mean_elite_reward:
                counter_for_restart += 1
            else:
                counter_for_restart = 0
            if counter_for_restart >= self.restart_timeout and self.restart_timeout > 0:
                population = self.init_population()
                counter_for_restart = 0
                self._log('Restarted population.')
            else:
                new_indices = torch.ones(elite_population.size(0), device=self.device).multinomial(
                    self.population_size, replacement=True)
                if self.reduce_to_mean:
                    elite_population = torch.mean(elite_population, dim=0).repeat(self.n_elite, 1)
                new_population = []
                for index in new_indices:
                    sample = elite_population[index]
                    new_population.append(
                        sample + self.sigma * torch.randn_like(sample, device=self.device)
                    )
                population = torch.stack(new_population)
            prev_mean_elite_reward = mean_elite_reward.cpu().numpy()
        
        if not self.converged:
            if self.output_handler is not None:
                t = time.time() - self.start_time
                self.output_handler.save_search_info(t, self.num_evaluations, self.best_reward,
                                                     self.best_program)
        
        return self.best_program, self.converged, self.num_evaluations
