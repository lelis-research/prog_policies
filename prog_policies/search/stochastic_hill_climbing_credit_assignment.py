import copy

import numpy as np

from prog_policies.base import dsl_nodes

from .stochastic_hill_climbing import StochasticHillClimbing
from .utils import evaluate_and_assign_credit

class StochasticHillClimbingWithCreditAssignment(StochasticHillClimbing):
    
    def init_search_vars(self):
        self.current_program = self.random_program()
        self.current_reward, self.current_nodes_score = evaluate_and_assign_credit(self.current_program, self.dsl, self.task_envs)
    
    def get_search_vars(self) -> dict:
        return {
            'current_program': self.current_program,
            'current_reward': self.current_reward,
            'current_nodes_score': self.current_nodes_score,
        }
    
    def set_search_vars(self, search_vars: dict):
        self.current_program = search_vars.get('current_program')
        self.current_reward = search_vars.get('current_reward')
        self.current_nodes_score = search_vars.get('current_nodes_score')
        
    def softmax(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def mutate_current_program(self) -> dsl_nodes.Program:
        mutated_program = copy.deepcopy(self.current_program)
        
        probs = self.softmax(-np.array(self.current_nodes_score[1:]))
        index = self.np_rng.choice(len(probs), p=probs) + 1
        node_to_mutate = mutated_program.get_all_nodes()[index]
        
        self.find_node_and_mutate(mutated_program, node_to_mutate)
        
        return mutated_program
    
    def search_iteration(self):
        if self.current_iteration % 100 == 0:
            self.log(f'Iteration {self.current_iteration}: Best reward {self.best_reward}, evaluations {self.num_evaluations}')
        
        if self.current_reward > self.best_reward:
            self.best_reward = self.current_reward
            self.best_program = self.dsl.parse_node_to_str(self.current_program)
            self.save_best()
        if self.best_reward >= 1.0:
            return
        
        next_program = self.mutate_current_program()
        next_reward, next_nodes_score = evaluate_and_assign_credit(next_program, self.dsl, self.task_envs)
        self.num_evaluations += 1
        
        if next_reward >= self.current_reward:
            self.current_program = next_program
            self.current_reward = next_reward
            self.current_nodes_score = next_nodes_score