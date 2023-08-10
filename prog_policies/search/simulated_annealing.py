from __future__ import annotations

import numpy as np

from .stochastic_hill_climbing import StochasticHillClimbing
from .utils import evaluate_program

class SimulatedAnnealing(StochasticHillClimbing):
    """Simulated Annealing in programmatic space
    """
    def parse_method_args(self, search_method_args: dict):
        self.initial_temperature = search_method_args.get('initial_temperature', 100)
        self.alpha = search_method_args.get('alpha', 0.9)
        self.beta = search_method_args.get('beta', 200)
        
    def init_search_vars(self):
        self.iterations_since_restart = 0
        self.current_program = self.random_program()
        self.current_reward = evaluate_program(self.current_program, self.dsl, self.task_envs)
        self.current_temperature = self.initial_temperature
        
    def get_search_vars(self) -> dict:
        return {
            'iterations_since_restart': self.iterations_since_restart,
            'current_program': self.current_program,
            'current_reward': self.current_reward,
            'current_temperature': self.current_temperature
        }
    
    def set_search_vars(self, search_vars: dict):
        self.iterations_since_restart = search_vars.get('iterations_since_restart')
        self.current_program = search_vars.get('current_program')
        self.current_reward = search_vars.get('current_reward')
        self.current_temperature = search_vars.get('current_temperature')
        
    def accept_function(self, current_r: float, next_r: float) -> float:
        return np.exp(self.beta * (next_r - current_r) / self.current_temperature)
    
    def decrease_temperature(self, i: int) -> None:
        self.current_temperature = self.initial_temperature / (1 + self.alpha * i)
    
    def search_iteration(self):
        if self.current_iteration % 100 == 0:
            self.log(f'Iteration {self.current_iteration}: Best reward {self.best_reward}, temperature {self.current_temperature}, evaluations {self.num_evaluations}')
        
        if self.current_reward > self.best_reward:
            self.best_reward = self.current_reward
            self.best_program = self.dsl.parse_node_to_str(self.current_program)
            self.save_best()
        if self.best_reward >= 1.0:
            return
        
        if self.current_temperature > 1.0:
            next_program = self.mutate_current_program()
            next_reward = evaluate_program(next_program, self.dsl, self.task_envs)
            self.num_evaluations += 1
            
            if next_reward > self.best_reward:
                self.best_reward = next_reward
                self.best_program = self.dsl.parse_node_to_str(next_program)
                self.save_best()
            
            if self.np_rng.rand() < self.accept_function(self.current_reward, next_reward):
                self.current_program = next_program
                self.current_reward = next_reward
                
            self.iterations_since_restart += 1
            self.decrease_temperature(self.iterations_since_restart)
            
        else:
            if self.best_reward > 0.0:
                self.current_program = self.dsl.parse_str_to_node(self.best_program)

            else:
                self.current_program = self.random_program()
                self.current_reward = evaluate_program(self.current_program, self.dsl, self.task_envs)
                self.num_evaluations += 1

            self.current_temperature = self.initial_temperature
            self.iterations_since_restart = 0
