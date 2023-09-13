from __future__ import annotations
import copy

from prog_policies.base import dsl_nodes

from .simulated_annealing import SimulatedAnnealing

class SimulatedAnnealingWithConstraint(SimulatedAnnealing):
    
    def parse_method_args(self, search_method_args: dict):
        self.size_constraint = search_method_args.get('size_constraint', 20)
        return super().parse_method_args(search_method_args)
    
    def mutate_current_program(self) -> dsl_nodes.Program:
        accepted = False
        while not accepted:
            mutated_program = copy.deepcopy(self.current_program)
        
            node_to_mutate = self.np_rng.choice(mutated_program.get_all_nodes()[1:])
            self.mutate_node(node_to_mutate)
            
            if mutated_program.get_size() <= self.size_constraint:
                accepted = True
        
        return mutated_program