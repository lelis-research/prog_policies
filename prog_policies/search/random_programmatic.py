from __future__ import annotations
import copy

import numpy as np

from prog_policies.base import dsl_nodes

from .base_search import BaseSearch
from .utils import evaluate_program

class RandomProgrammatic(BaseSearch):
    def parse_method_args(self, search_method_args: dict):
        self.max_depth = search_method_args.get('max_depth', 6)
        self.max_sequence = search_method_args.get('max_sequence', 8)
        
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
        self.fill_children_of_node(program, max_depth=self.max_depth, max_sequence=self.max_sequence)
        return program
    
    def search_iteration(self):
        self.log(f'Iteration {self.current_iteration}: Best reward {self.best_reward}, evaluations {self.num_evaluations}')
        
        program = self.random_program()
        reward = evaluate_program(program, self.dsl, self.task_envs)
        self.num_evaluations += 1
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_program = self.dsl.parse_node_to_str(program)
            self.save_best()
