from __future__ import annotations
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import sys
from multiprocessing import Pool

sys.path.append('.')

from prog_policies.karel import KarelDSL
from prog_policies.karel_tasks import get_task_cls
from prog_policies.search_space import ProgrammaticSpace, LatentSpace
from prog_policies.search_methods import HillClimbing, CEM, CEBS


if __name__ == '__main__':
    
    n_env = 32
    n_search_iterations = 1000
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('task', help='Name of the task class')
    parser.add_argument('k', type=int, help='Number of neighbors to consider')
    parser.add_argument('e', type=int, help='Number of elite candidates in CEM-based methods')
    parser.add_argument('space', help='Search space to use')
    parser.add_argument('seed_lb', type=int, help='Lower bound for seed')
    parser.add_argument('seed_ub', type=int, help='Upper bound for seed')
    parser.add_argument('method', help='Search method to use')
    
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
        'latent': LatentSpace(dsl, sigma)
    }
    
    env_args = {
        "env_height": 8,
        "env_width": 8,
        "crashable": False,
        "leaps_behaviour": True,
        "max_calls": 10000
    }
    
    search_methods = {
        'hc': HillClimbing(args.k),
        'cem': CEM(args.k, args.e),
        'cebs': CEBS(args.k, args.e)
    }
    
    search_method = search_methods[args.method]
    
    if args.task == "CleanHouse":
        env_args["env_height"] = 14
        env_args["env_width"] = 22
    
    fname = f'output/{args.method}_rewards_{args.space}_k{args.k}_{args.task}_seeds{args.seed_lb}-{args.seed_ub}.csv'

    task_cls = get_task_cls(args.task)
    task_envs = [task_cls(env_args, i) for i in range(n_env)]
    
    with open(fname, 'w') as f:
        f.write('')
    def f(seed):
        r = search_method.search(search_spaces[args.space], task_envs, seed, n_search_iterations)
        with open(fname, 'a') as f:
            f.write(','.join([str(reward) for reward in r]) + '\n')
        return r
    with Pool() as pool:
        pool.map(f, range(args.seed_lb, args.seed_ub))
