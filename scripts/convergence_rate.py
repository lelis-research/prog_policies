from __future__ import annotations
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import sys
from multiprocessing import Pool

sys.path.append('.')

from prog_policies.base import dsl_nodes, BaseTask
from prog_policies.karel import KarelDSL
from prog_policies.karel_tasks import get_task_cls
from prog_policies.search_space import BaseSearchSpace, ProgrammaticSpace, LatentSpace

def evaluate_program(program: dsl_nodes.Program, task_envs: list[BaseTask]) -> float:
    sum_reward = 0.
    for task_env in task_envs:
        sum_reward += task_env.evaluate_program(program)
    return sum_reward / len(task_envs)

def stochastic_hill_climbing(search_space: BaseSearchSpace, task_envs: list[BaseTask],
                             seed = None, n_iterations: int = 1000) -> float:
    search_space.initialize_program(seed)
    current_program = search_space.get_current_program()
    best_reward = evaluate_program(current_program, task_envs)
    for _ in range(n_iterations):
        search_space.mutate_current_program()
        program = search_space.get_current_program()
        reward = evaluate_program(program, task_envs)
        if reward >= best_reward:
            best_reward = reward
        else:
            search_space.rollback_mutation()
    return best_reward

if __name__ == '__main__':
    
    n_env = 32
    n_search_iterations = 1000
    n_tries = 100
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('task', help='Name of the task class')
    
    args = parser.parse_args()
    
    dsl = KarelDSL()
    search_spaces = [
        ProgrammaticSpace(dsl),
        LatentSpace(dsl)
    ]
    search_spaces_labels = [
        'programmatic',
        'latent'
    ]
    
    env_args = {
        "env_height": 8,
        "env_width": 8,
        "crashable": False,
        "leaps_behaviour": True,
        "max_calls": 10000
    }
    
    if args.task == "CleanHouse":
        env_args["env_height"] = 14
        env_args["env_width"] = 22
    
    task_cls = get_task_cls(args.task)
    task_envs = [task_cls(env_args, i) for i in range(n_env)]
    
    for search_space, search_space_label in zip(search_spaces, search_spaces_labels):
        os.makedirs(f'output/rewards_{search_space_label}_{args.task}', exist_ok=True)
        def f(seed):
            r = stochastic_hill_climbing(search_space, task_envs, seed, n_search_iterations)
            with open(f'output/rewards_{search_space_label}_{args.task}/{seed}.csv', 'w') as f:
                f.write(str(r))
        with Pool() as pool:
            pool.map(f, range(n_tries))
        
