from __future__ import annotations
from typing import Union

from prog_policies.base import BaseTask, BaseDSL, dsl_nodes

def evaluate_program(program: Union[str, dsl_nodes.Program], dsl: BaseDSL,
                     task_envs: list[BaseTask]) -> float:
    if isinstance(program, str):
        try:
            program = dsl.parse_str_to_node(program)
        except AssertionError: # In case of invalid program (e.g. does not have an ending token)
            return -float('inf')
    
    sum_reward = 0.
    for task_env in task_envs:
        sum_reward += task_env.evaluate_program(program)
    
    return sum_reward / len(task_envs)
