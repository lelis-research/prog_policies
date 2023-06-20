from __future__ import annotations
import copy

import torch

from prog_policies.base import BaseDSL, BaseTask, dsl_nodes
from prog_policies.latent_space.models import BaseVAE
from prog_policies.output_handler import OutputHandler
from .simulated_annealing import SimulatedAnnealing

class LatentSimulatedAnnealing(SimulatedAnnealing):
    
    def __init__(self, dsl: BaseDSL, task_cls: type[BaseTask], model: BaseVAE,
                 number_executions: int = 16, env_args: dict = {}, sigma: float = 0.1,
                 initial_temperature: float = 100, alpha: float = 0.9,
                 beta: float = 200, number_iterations: int = 1000,
                 seed: int = None, output_handler: OutputHandler = None):
        super().__init__(dsl, task_cls, number_executions, env_args,
                         initial_temperature, alpha, beta, number_iterations,
                         seed, output_handler)
        self.model = model
        self.sigma = sigma
        
    def random_program(self) -> dsl_nodes.Program:
        latent = self.rng.randn(1, self.model.hidden_size)
        program_tokens = self.model.decode_vector(
            torch.tensor(latent, dtype=torch.float32, device=self.model.device)
        )[0]
        return self.dsl.parse_int_to_node(program_tokens)
    
    def find_and_mutate(self, node: dsl_nodes.BaseNode, index_to_mutate: int) -> bool:
        for i, child_type in enumerate(node.children_types):
            if index_to_mutate == self.current_index:
                if child_type == dsl_nodes.StatementNode:
                    current_child_tokens = self.dsl.parse_node_to_int(node.children[i])
                    current_child_tokens = [self.dsl.t2i['DEF'], self.dsl.t2i['run'], self.dsl.t2i['m(']] + current_child_tokens + [self.dsl.t2i['m)']]
                    current_latent = self.model.encode_program(
                        torch.tensor(current_child_tokens, dtype=torch.long, device=self.model.device)
                    ).detach().cpu().numpy()
                    new_latent = current_latent + self.rng.randn(1, self.model.hidden_size) * self.sigma
                    new_child_tokens = self.model.decode_vector(
                        torch.tensor(new_latent, dtype=torch.float32, device=self.model.device)
                    )[0]
                    try:
                        new_child = self.dsl.parse_int_to_node(new_child_tokens)
                    except AssertionError:
                        return False
                    node.children[i] = new_child.children[0]
                    return True
                else:
                    return False
            else:
                self.current_index += 1
                return self.find_and_mutate(node.children[i], index_to_mutate)
    
    def mutate(self, program: dsl_nodes.Program) -> dsl_nodes.Program:
        copy_program = copy.deepcopy(program)
        
        success = False
        while not success:
            index = self.rng.randint(copy_program.get_size())
            self.current_index = 0
            success = self.find_and_mutate(copy_program, index)
            if self.dsl.parse_node_to_str(copy_program) == self.dsl.parse_node_to_str(program):
                success = False
        
        return copy_program