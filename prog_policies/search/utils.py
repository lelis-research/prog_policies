from __future__ import annotations
from typing import Union
import copy

from prog_policies.base import BaseTask, BaseDSL, dsl_nodes

def simplify_program(program: dsl_nodes.Program, nodes_count: dict) -> dsl_nodes.Program:
    simplified_program = copy.deepcopy(program)
    all_nodes = simplified_program.get_all_nodes()
    new_nodes_dict = {node: i for node, i in zip(all_nodes, nodes_count.values())}
    nodes_to_remove: set[dsl_nodes.BaseNode] = set()
    for node in all_nodes:
        if not isinstance(node, dsl_nodes.Program) \
                and issubclass(type(node), dsl_nodes.StatementNode):
            if new_nodes_dict[node] == 0. and new_nodes_dict[node.parent] > 0.:
                nodes_to_remove.add(node)
    for node in nodes_to_remove:
        if isinstance(node.parent, dsl_nodes.Concatenate):
            if node.parent.children[0] == node:
                for i, child in enumerate(node.parent.parent.children):
                    if child == node.parent:
                        node.parent.parent.children[i] = node.parent.children[1]
                        node.parent.children[1].parent = node.parent.parent
            else:
                for i, child in enumerate(node.parent.parent.children):
                    if child == node.parent:
                        node.parent.parent.children[i] = node.parent.children[0]
                        node.parent.children[0].parent = node.parent.parent
        elif isinstance(node.parent, dsl_nodes.If):
            pass # TODO
        elif isinstance(node.parent, dsl_nodes.ITE):
            if node.parent.children[1] == node:
                for i, child in enumerate(node.parent.parent.children):
                    if child == node.parent:
                        not_cond = dsl_nodes.Not.new(node.parent.children[0])
                        new_node = dsl_nodes.If.new(not_cond, node.parent.children[2])
                        node.parent.parent.children[i] = new_node
                        new_node.parent = node.parent.parent
            else:
                for i, child in enumerate(node.parent.parent.children):
                    if child == node.parent:
                        new_node = dsl_nodes.If.new(node.parent.children[0], node.parent.children[1])
                        node.parent.parent.children[i] = new_node
                        new_node.parent = node.parent.parent
        elif isinstance(node.parent, dsl_nodes.While):
            pass # TODO
    return simplified_program

def evaluate_program(program: Union[str, dsl_nodes.Program], dsl: BaseDSL,
                     task_envs: list[BaseTask]) -> float:
    if isinstance(program, str):
        try:
            program = dsl.parse_str_to_node(program)
        except Exception: # In case of invalid program (e.g. does not have an ending token)
            return -float('inf')
    
    sum_reward = 0.
    for task_env in task_envs:
        sum_reward += task_env.evaluate_program(program)
    
    return sum_reward / len(task_envs)

def evaluate_and_assign_credit(program: Union[str, dsl_nodes.Program], dsl: BaseDSL,
                               task_envs: list[BaseTask]) -> tuple[float, dict, dict]:
    if isinstance(program, str):
        try:
            program = dsl.parse_str_to_node(program)
        except Exception: # In case of invalid program (e.g. does not have an ending token)
            return -float('inf')
    
    mean_reward = 0.
    all_nodes = program.get_all_nodes()
    mean_node_score = {node: 0. for node in all_nodes}
    mean_node_count = {node: 0 for node in all_nodes}
    for task_env in task_envs:
        reward, node_score, node_count = task_env.evaluate_and_assign_credit(program)
        mean_reward += reward / len(task_envs)
        for node in all_nodes:
            mean_node_score[node] += node_score[node] / len(task_envs)
            mean_node_count[node] += node_count[node] / len(task_envs)
    return mean_reward, mean_node_score, mean_node_count