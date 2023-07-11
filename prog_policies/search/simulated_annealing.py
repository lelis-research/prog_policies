from __future__ import annotations
import copy

import numpy as np

from prog_policies.base import dsl_nodes

from .base_search import BaseSearch
from .utils import evaluate_program

class SimulatedAnnealing(BaseSearch):
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
        
    def fill_children_of_node(self, node: dsl_nodes.BaseNode,
                          current_depth: int = 1, current_sequence: int = 0,
                          max_depth: int = 4, max_sequence: int = 6) -> None:
        node_prod_rules = self.dsl.prod_rules[type(node)]
        for i, child_type in enumerate(node.get_children_types()):
            child_probs = self.dsl.get_dsl_nodes_probs(child_type)
            for child_type in child_probs:
                if child_type not in node_prod_rules[i]:
                    child_probs[child_type] = 0.
                if current_depth >= max_depth and child_type.get_node_depth() > 0:
                    child_probs[child_type] = 0.
            if issubclass(type(node), dsl_nodes.Concatenate) and current_sequence + 1 >= max_sequence:
                if dsl_nodes.Concatenate in child_probs:
                    child_probs[dsl_nodes.Concatenate] = 0.
            
            p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
            child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
            child_instance = child()
            if child.get_number_children() > 0:
                if issubclass(type(node), dsl_nodes.Concatenate):
                    self.fill_children_of_node(child_instance, current_depth + child.get_node_depth(),
                                               current_sequence + 1, max_depth, max_sequence)
                else:
                    self.fill_children_of_node(child_instance, current_depth + child.get_node_depth(),
                                               1, max_depth, max_sequence)
            
            elif isinstance(child_instance, dsl_nodes.Action):
                child_instance.name = self.np_rng.choice(list(self.dsl.action_probs.keys()),
                                                         p=list(self.dsl.action_probs.values()))
            elif isinstance(child_instance, dsl_nodes.BoolFeature):
                child_instance.name = self.np_rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                         p=list(self.dsl.bool_feat_probs.values()))
            elif isinstance(child_instance, dsl_nodes.ConstInt):
                child_instance.value = self.np_rng.choice(list(self.dsl.const_int_probs.keys()),
                                                          p=list(self.dsl.const_int_probs.values()))
            node.children[i] = child_instance
    
    def random_program(self) -> dsl_nodes.Program:
        program = dsl_nodes.Program()
        self.fill_children_of_node(program, max_depth=4, max_sequence=6)
        return program
    
    def find_and_mutate(self, node: dsl_nodes.BaseNode, index_to_mutate: int) -> None:
        for i, child_type in enumerate(node.children_types):
            if self.current_index == index_to_mutate:
                node_prod_rules = self.dsl.prod_rules[type(node)]
                child_probs = self.dsl.get_dsl_nodes_probs(child_type)
                for child_type in child_probs:
                    if child_type not in node_prod_rules[i]:
                        child_probs[child_type] = 0.
                
                p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
                child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
                child_instance = child()
                if child.get_number_children() > 0:
                    self.fill_children_of_node(child_instance, max_depth=2, max_sequence=4)
                elif isinstance(child_instance, dsl_nodes.Action):
                    child_instance.name = self.np_rng.choice(list(self.dsl.action_probs.keys()),
                                                             p=list(self.dsl.action_probs.values()))
                elif isinstance(child_instance, dsl_nodes.BoolFeature):
                    child_instance.name = self.np_rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                             p=list(self.dsl.bool_feat_probs.values()))
                elif isinstance(child_instance, dsl_nodes.ConstInt):
                    child_instance.value = self.np_rng.choice(list(self.dsl.const_int_probs.keys()),
                                                              p=list(self.dsl.const_int_probs.values()))
                node.children[i] = child_instance
                return
            else:
                self.current_index += 1
                self.find_and_mutate(node.children[i], index_to_mutate)
    
    def mutate_current_program(self) -> dsl_nodes.Program:
        mutated_program = copy.deepcopy(self.current_program)
        
        index = self.np_rng.randint(mutated_program.get_size())
        self.current_index = 0
        self.find_and_mutate(mutated_program, index)
        
        return mutated_program
    
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
