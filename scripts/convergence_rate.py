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
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('task', help='Name of the task class')
    parser.add_argument('k', type=int, help='Number of neighbors to consider')
    parser.add_argument('space', help='Search space to use')
    parser.add_argument('seed_lb', type=int, help='Lower bound for seed')
    parser.add_argument('seed_ub', type=int, help='Upper bound for seed')
    
    args = parser.parse_args()

    if args.task == "CleanHouse" or args.task == "StairClimberSparse" or args.task == "TopOff":
        sigma = 0.25
    elif args.task == "FourCorners" or args.task == "Harvester":
        sigma = 0.5
    elif args.task == "MazeSparse":
        sigma = 0.1
    else:
        sigma = 0.25
    
    dsl = KarelDSL()
    
    search_spaces = {
        'programmatic': ProgrammaticSpace(dsl, sigma),
        'latent': LatentSpace(dsl, sigma),
        'latent2': LatentSpace2(dsl, sigma)
    }
    
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
    
    fname = f'output/rewards_{args.space}_k{args.k}_{args.task}_seeds{args.seed_lb}-{args.seed_ub}.csv'

    task_cls = get_task_cls(args.task)
    task_envs = [task_cls(env_args, i) for i in range(n_env)]
    
    with open(fname, 'w') as f:
        f.write('')
    def f(seed):
        r = stochastic_hill_climbing(search_spaces[args.space], task_envs, seed, n_search_iterations, args.k)
        with open(fname, 'a') as f:
            f.write(','.join([str(reward) for reward in r]) + '\n')
        return r
    with Pool() as pool:
        pool.map(f, range(args.seed_lb, args.seed_ub))
