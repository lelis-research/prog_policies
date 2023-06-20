from __future__ import annotations
import copy
import time
import numpy as np

from prog_policies.base import BaseDSL, BaseTask, dsl_nodes
from prog_policies.output_handler import OutputHandler

from .utils import evaluate_program

class SimulatedAnnealing:
    
    def __init__(self, dsl: BaseDSL, task_cls: type[BaseTask], number_executions: int = 16,
                 env_args: dict = {}, initial_temperature: float = 100.0, alpha: float = 0.9,
                 beta: float = 200.0, number_iterations: int = 1000, seed: int = None,
                 output_handler: OutputHandler = None):
        self.dsl = dsl
        self.task_envs = [task_cls(env_args, i) for i in range(number_executions)]
        self.initial_temperature = initial_temperature
        self.alpha = alpha
        self.beta = beta
        self.number_iterations = number_iterations
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        self.output_handler = output_handler
        if self.output_handler is not None:
            self.output_handler.setup_search()
            self.output_handler.setup_search_info(seed, task_cls.__name__)
    
    def _log(self, message: str):
        if self.output_handler is not None:
            self.output_handler.log('Simulated Annealing', message)

    def _save_best(self):
        if self.output_handler is not None:
            t = time.time() - self.start_time
            self.output_handler.save_search_info(t, self.num_evaluations, self.best_reward,
                                                    self.best_program_str)
    
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
            child = self.rng.choice(list(child_probs.keys()), p=p_list)
            child_instance = child()
            if child.get_number_children() > 0:
                if issubclass(type(node), dsl_nodes.Concatenate):
                    self.fill_children_of_node(child_instance, current_depth + child.get_node_depth(),
                                               current_sequence + 1, max_depth, max_sequence)
                else:
                    self.fill_children_of_node(child_instance, current_depth + child.get_node_depth(),
                                               1, max_depth, max_sequence)
            
            elif isinstance(child_instance, dsl_nodes.Action):
                child_instance.name = self.rng.choice(list(self.dsl.action_probs.keys()),
                                                      p=list(self.dsl.action_probs.values()))
            elif isinstance(child_instance, dsl_nodes.BoolFeature):
                child_instance.name = self.rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                      p=list(self.dsl.bool_feat_probs.values()))
            elif isinstance(child_instance, dsl_nodes.ConstInt):
                child_instance.value = self.rng.choice(list(self.dsl.const_int_probs.keys()),
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
                child = self.rng.choice(list(child_probs.keys()), p=p_list)
                child_instance = child()
                if child.get_number_children() > 0:
                    self.fill_children_of_node(child_instance, max_depth=2, max_sequence=4)
                elif isinstance(child_instance, dsl_nodes.Action):
                    child_instance.name = self.rng.choice(list(self.dsl.action_probs.keys()),
                                                         p=list(self.dsl.action_probs.values()))
                elif isinstance(child_instance, dsl_nodes.BoolFeature):
                    child_instance.name = self.rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                          p=list(self.dsl.bool_feat_probs.values()))
                elif isinstance(child_instance, dsl_nodes.ConstInt):
                    child_instance.value = self.rng.choice(list(self.dsl.const_int_probs.keys()),
                                                           p=list(self.dsl.const_int_probs.values()))
                node.children[i] = child_instance
                return
            else:
                self.current_index += 1
                self.find_and_mutate(node.children[i], index_to_mutate)
    
    def mutate(self, program: dsl_nodes.Program) -> dsl_nodes.Program:
        copy_program = copy.deepcopy(program)
        
        index = self.rng.randint(copy_program.get_size())
        self.current_index = 0
        self.find_and_mutate(copy_program, index)
        
        return copy_program
    
    def accept_function(self, current_r: float, next_r: float) -> float:
        return np.exp(self.beta * (next_r - current_r) / self.current_temperature)
    
    def decrease_temperature(self, i: int) -> None:
        self.current_temperature = self.initial_temperature / (1 + self.alpha * i)
        
    def search(self) -> tuple[dsl_nodes.Program, int, float]:
        self.best_reward = -float('inf')
        self.best_program = None
        self.best_program_str = None
        self.start_time = time.time()
        
        self.num_evaluations = 0
        current_program = self.random_program()

        for i in range(1, self.number_iterations + 1):
            
            self._log(f'Iteration {i}, evaluations so far: {self.num_evaluations}')
            self.current_temperature = self.initial_temperature

            current_r = evaluate_program(current_program, self.dsl, self.task_envs)
            self.num_evaluations += 1
            if current_r > self.best_reward:
                self.best_reward = current_r
                self.best_program = current_program
                self.best_program_str = self.dsl.parse_node_to_str(current_program)
                self._log(f'New best reward: {self.best_reward}')
                self._log(f'New best program: {self.best_program_str}')
                self._log(f'Number of evaluations: {self.num_evaluations}')
                self._save_best()
            if self.best_reward == 1:
                return self.best_program_str, self.num_evaluations, self.best_reward
            
            iteration_number = 1
            while self.current_temperature > 1.0:
                
                next_program = self.mutate(current_program)
                
                next_r = evaluate_program(next_program, self.dsl, self.task_envs)
                self.num_evaluations += 1
                if next_r > self.best_reward:
                    self.best_reward = next_r
                    self.best_program = next_program
                    self.best_program_str = self.dsl.parse_node_to_str(next_program)
                    self._log(f'New best reward: {self.best_reward}')
                    self._log(f'New best program: {self.best_program_str}')
                    self._log(f'Number of evaluations: {self.num_evaluations}')
                    self._save_best()
                if self.best_reward == 1:
                    return self.best_program_str, self.num_evaluations, self.best_reward
                
                if self.rng.rand() < self.accept_function(current_r, next_r):
                    current_program = next_program
                    current_r = next_r
                
                iteration_number += 1
                self.decrease_temperature(iteration_number)
                
            if self.best_reward <= 0.0:
                current_program = self.random_program()
            else:
                current_program = self.best_program
                
        return self.best_program_str, self.num_evaluations, self.best_reward
            