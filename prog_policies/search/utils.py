from __future__ import annotations
from typing import Union

from prog_policies.base import BaseTask, BaseDSL, dsl_nodes

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
                               task_envs: list[BaseTask]) -> tuple[float, list[float]]:
    if isinstance(program, str):
        try:
            program = dsl.parse_str_to_node(program)
        except Exception: # In case of invalid program (e.g. does not have an ending token)
            return -float('inf')
    
    mean_reward = 0.
    mean_node_score = [0. for _ in range(len(program.get_all_nodes()))]
    for task_env in task_envs:
        reward, node_score = task_env.evaluate_and_assign_credit(program)
        mean_reward += reward / len(task_envs)
        for i, score in enumerate(node_score):
            mean_node_score[i] += score / len(task_envs)
    return mean_reward, mean_node_score