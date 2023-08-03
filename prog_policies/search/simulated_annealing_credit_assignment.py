import copy

import numpy as np

from prog_policies.base import dsl_nodes

from .simulated_annealing import SimulatedAnnealing
from .utils import evaluate_and_assign_credit

class SimulatedAnnealingWithCreditAssignment(SimulatedAnnealing):
    
    def init_search_vars(self):
        self.iterations_since_restart = 0
        self.current_program = self.random_program()
        self.current_reward, self.current_nodes_score = evaluate_and_assign_credit(self.current_program, self.dsl, self.task_envs)
        self.current_temperature = self.initial_temperature
        self.best_nodes_score = self.current_nodes_score
    
    def get_search_vars(self) -> dict:
        return {
            'iterations_since_restart': self.iterations_since_restart,
            'current_program': self.current_program,
            'current_reward': self.current_reward,
            'current_nodes_score': self.current_nodes_score,
            'current_temperature': self.current_temperature,
            'best_nodes_score': self.best_nodes_score
        }
    
    def set_search_vars(self, search_vars: dict):
        self.iterations_since_restart = search_vars.get('iterations_since_restart')
        self.current_program = search_vars.get('current_program')
        self.current_reward = search_vars.get('current_reward')
        self.current_nodes_score = search_vars.get('current_nodes_score')
        self.current_temperature = search_vars.get('current_temperature')
        self.best_nodes_score = search_vars.get('best_nodes_score')
        
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
            self.log(f'Iteration {self.current_iteration}: Best reward {self.best_reward}, temperature {self.current_temperature}, evaluations {self.num_evaluations}')
        
        if self.current_reward > self.best_reward:
            self.best_reward = self.current_reward
            self.best_program = self.dsl.parse_node_to_str(self.current_program)
            self.best_nodes_score = self.current_nodes_score
            self.save_best()
        if self.best_reward >= 1.0:
            return
        
        if self.current_temperature > 1.0:
            next_program = self.mutate_current_program()
            next_reward, next_nodes_score = evaluate_and_assign_credit(next_program, self.dsl, self.task_envs)
            self.num_evaluations += 1
            
            if next_reward > self.best_reward:
                self.best_reward = next_reward
                self.best_program = self.dsl.parse_node_to_str(next_program)
                self.best_nodes_score = next_nodes_score
                self.save_best()
            
            if self.np_rng.rand() < self.accept_function(self.current_reward, next_reward):
                self.current_program = next_program
                self.current_reward = next_reward
                self.current_nodes_score = next_nodes_score
                
            self.iterations_since_restart += 1
            self.decrease_temperature(self.iterations_since_restart)
            
        else:
            if self.best_reward > 0.0:
                self.current_program = self.dsl.parse_str_to_node(self.best_program)
                self.current_reward = self.best_reward
                self.current_nodes_score = self.best_nodes_score
            else:
                self.current_program = self.random_program()
                self.current_reward, self.current_nodes_score = evaluate_and_assign_credit(self.current_program, self.dsl, self.task_envs)
                self.num_evaluations += 1

            self.current_temperature = self.initial_temperature
            self.iterations_since_restart = 0