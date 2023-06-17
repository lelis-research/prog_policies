from __future__ import annotations
import copy

from prog_policies.base import BaseDSL, BaseTask, dsl_nodes

from .utils import evaluate_program

class TopDownSearch:
    
    def __init__(self, dsl: BaseDSL, task_cls: type[BaseTask], number_executions: int = 16,
                 env_args: dict = {}) -> None:
        self.dsl = dsl
        self.task_envs = [task_cls(i, env_args) for i in range(number_executions)]

    def get_number_holes(self, node: dsl_nodes.BaseNode) -> int:
        holes = 0
        for child in node.children:
            if child is None:
                holes += 1
            else:
                holes += self.get_number_holes(child)
        return holes
    
    def grow_node(self, node: dsl_nodes.BaseNode, grow_bound: int) -> list[dsl_nodes.BaseNode]:
        n_holes = self.get_number_holes(node)
        if n_holes == 0:
            return []

        grown_children = []
        prod_rule = self.dsl.prod_rules[type(node)]
        for i, child in enumerate(node.children):
            if child is None:
                # Replace child with possible production rules
                child_list = [n for n in self.dsl.nodes_list
                              if type(n) in prod_rule[i]
                              and n.get_size() + len(n.children) <= grow_bound - n_holes + 1]
                grown_children.append(child_list)
            else:
                grown_children.append(self.grow_node(child, grow_bound))
                
        grown_nodes = []
        for i, grown_child in enumerate(grown_children):
            for c in grown_child:
                grown_node = type(node)()
                grown_node.children = copy.deepcopy(node.children)
                grown_node.children[i] = c
                grown_nodes.append(grown_node)
        return grown_nodes
    
    def search(self, initial_program: dsl_nodes.Program, grow_bound: int = 5) -> tuple[dsl_nodes.Program, int, float]:
        self.best_reward = -float('inf')
        self.best_program = None
        
        num_evaluations = 0
        plist = [initial_program]

        for i in range(grow_bound):
            # Grow programs once
            new_plist = []
            for p in plist:
                new_plist += self.grow_node(p, grow_bound - i)
            plist = new_plist
            # Evaluate programs
            complete_programs = [p for p in plist if p.is_complete()]
            for p in complete_programs:
                r = evaluate_program(p, self.task_envs, self.best_reward)
                num_evaluations += 1
                if r > self.best_reward:
                    self.best_reward = r
                    self.best_program = p
                if self.best_reward == 1:
                    return self.best_program, num_evaluations, self.best_reward

        return self.best_program, num_evaluations, self.best_reward