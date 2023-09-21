import copy
import numpy as np

from ..base.dsl import dsl_nodes

from .base_space import BaseSearchSpace

# recursively calculate the node depth (number of levels from root)
def get_max_depth(program: dsl_nodes.Program) -> int:
    depth = 0
    for child in program.children:
        if child is not None:
            depth = max(depth, get_max_depth(child))
    return depth + program.node_depth
    
# recursively calculate the max number of joined concatenate nodes
def get_max_sequence(program: dsl_nodes.Program) -> int:
    max_sequence = 0
    # TODO
    # for child in program.children:
    #     if issubclass(type(child), dsl_nodes.Concatenate):
    #         max_sequence = max(max_sequence, child.get_max_sequence())
    # if issubclass(type(self), Concatenate):
    #     max_sequence += 1
    return max_sequence

class ProgrammaticSpace(BaseSearchSpace):
    
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
            child_instance.parent = node

    def random_program(self):
        program = dsl_nodes.Program()
        self.fill_children_of_node(program, max_depth=4, max_sequence=6)
        return program
    
    def mutate_node(self, node_to_mutate: dsl_nodes.BaseNode) -> None:
        for i, child in enumerate(node_to_mutate.parent.children):
            if child == node_to_mutate:
                child_type = node_to_mutate.parent.children_types[i]
                node_prod_rules = self.dsl.prod_rules[type(node_to_mutate.parent)]
                child_probs = self.dsl.get_dsl_nodes_probs(child_type)
                for child_type in child_probs:
                    if child_type not in node_prod_rules[i]:
                        child_probs[child_type] = 0.
                
                p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
                child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
                child_instance = child()
                if child.get_number_children() > 0:
                    self.fill_children_of_node(child_instance)
                elif isinstance(child_instance, dsl_nodes.Action):
                    child_instance.name = self.np_rng.choice(list(self.dsl.action_probs.keys()),
                                                                p=list(self.dsl.action_probs.values()))
                elif isinstance(child_instance, dsl_nodes.BoolFeature):
                    child_instance.name = self.np_rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                                p=list(self.dsl.bool_feat_probs.values()))
                elif isinstance(child_instance, dsl_nodes.ConstInt):
                    child_instance.value = self.np_rng.choice(list(self.dsl.const_int_probs.keys()),
                                                                p=list(self.dsl.const_int_probs.values()))
                node_to_mutate.parent.children[i] = child_instance
                child_instance.parent = node_to_mutate.parent
    
    def mutate_current_program(self) -> None:
        accepted = False
        self.previous_program = self.current_program
        while not accepted:
            mutated_program = copy.deepcopy(self.current_program)
            node_to_mutate = self.np_rng.choice(mutated_program.get_all_nodes()[1:])
            self.mutate_node(node_to_mutate)
            accepted = get_max_depth(mutated_program) <= 4 and get_max_sequence(mutated_program) <= 6
        self.current_program = mutated_program
        
    def rollback_mutation(self):
        self.current_program = self.previous_program