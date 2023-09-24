from __future__ import annotations
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import sys
from multiprocessing import Pool

sys.path.append('.')

from prog_policies.base import dsl_nodes, BaseTask
from prog_policies.karel import KarelDSL
from prog_policies.karel_tasks import get_task_cls
from prog_policies.search_space import BaseSearchSpace, ProgrammaticSpace, LatentSpace, LatentSpace2

def evaluate_program(program: dsl_nodes.Program, task_envs: list[BaseTask]) -> float:
    sum_reward = 0.
    for task_env in task_envs:
        sum_reward += task_env.evaluate_program(program)
    return sum_reward / len(task_envs)

def stochastic_hill_climbing(search_space: BaseSearchSpace, task_envs: list[BaseTask],
                             seed = None, n_iterations: int = 10000, k: int = 5) -> list[float]:
    rewards = []
    search_space.set_seed(seed)
    best_ind, best_prog = search_space.initialize_individual()
    best_reward = evaluate_program(best_prog, task_envs)
    rewards.append(best_reward)
    for _ in range(n_iterations):
        candidates = search_space.get_neighbors(best_ind, k=k)
        in_local_maximum = True
        for ind, prog in candidates:
            reward = evaluate_program(prog, task_envs)
            if reward > best_reward:
                best_ind = ind
                best_prog = prog
                best_reward = reward
                in_local_maximum = False
                break
        if in_local_maximum: break
        rewards.append(best_reward)
    return rewards

if __name__ == '__main__':
    
    n_env = 32
    n_search_iterations = 1000
    n_tries = 100
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('task', help='Name of the task class')
    parser.add_argument('k', type=int, help='Number of neighbors to consider')
    
    args = parser.parse_args()

    if args.task == "CleanHouse" or args.task == "StairClimberSparse" or args.task == "TopOff":
        sigma = 0.25
    elif args.task == "FourCorner" or args.task == "Harvester":
        sigma = 0.5
    elif args.task == "MazeSparse":
        sigma = 0.1
    else:
        sigma = 0.25
    
    dsl = KarelDSL()
    search_spaces = [
        ProgrammaticSpace(dsl),
        LatentSpace(dsl, sigma),
        LatentSpace2(dsl, sigma),
    ]
    search_spaces_labels = [
        'programmatic',
        'latent',
        'latent2',
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
        os.makedirs(f'output/rewards_{search_space_label}_k{args.k}_{args.task}', exist_ok=True)
        def f(seed):
            r = stochastic_hill_climbing(search_space, task_envs, seed, n_search_iterations, args.k)
            with open(f'output/rewards_{search_space_label}_k{args.k}_{args.task}/{seed}.csv', 'w') as f:
                f.write(",".join(map(str, r)))
        with Pool() as pool:
            pool.map(f, range(n_tries))
        
