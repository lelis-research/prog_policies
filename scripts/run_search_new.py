import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.append('.')

from prog_policies.karel import KarelDSL
from prog_policies.karel_tasks import get_task_cls
from prog_policies.search_space import get_search_space_cls
from prog_policies.search_methods import get_search_method_cls

if __name__ == '__main__':
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    # parser.add_argument('--search_args_path', default='sample_args/search/latent_cem.json', help='Arguments path for search method')
    # parser.add_argument('--log_folder', default='logs', help='Folder to save logs')
    # parser.add_argument('--search_seed', type=int, help='Seed for search method')
    # parser.add_argument('--wandb_entity', type=str, help='Wandb entity')
    # parser.add_argument('--wandb_project', type=str, help='Wandb project')
    
    parser.add_argument('--search_space', default='ProgrammaticSpace', help='Search space class name')
    parser.add_argument('--search_method', default='HillClimbing', help='Search method class name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for searching')
    parser.add_argument('--num_iterations', type=int, default=1000000, help='Number of search iterations')
    parser.add_argument('--num_envs', type=int, default=32, help='Number of environments to search')
    parser.add_argument('--task', default='StairClimber', help='Task class name')
    parser.add_argument('--sigma', type=float, default=0.1, help='Standard deviation for Gaussian noise in Latent Space')
    parser.add_argument('--k', type=int, default=32, help='Number of neighbors to consider')
    parser.add_argument('--e', type=int, default=8, help='Number of elite candidates in CEM-based methods')
    
    args = parser.parse_args()
    
    dsl = KarelDSL()
    
    env_args = {
        "env_height": 8,
        "env_width": 8,
        "crashable": False,
        "leaps_behaviour": True,
        "max_calls": 10000
    }
    
    if args.task == "StairClimber" or args.task == "TopOff" or args.task == "FourCorners":
        env_args["env_height"] = 12
        env_args["env_width"] = 12
    
    if args.task == "CleanHouse":
        env_args["env_height"] = 14
        env_args["env_width"] = 22
    
    task_cls = get_task_cls(args.task)
    task_envs = [task_cls(env_args, i) for i in range(args.num_envs)]
    
    search_space_cls = get_search_space_cls(args.search_space)
    search_space = search_space_cls(dsl, args.sigma)
    search_space.set_seed(args.seed)
    
    search_method_cls = get_search_method_cls(args.search_method)
    search_method = search_method_cls(args.k, args.e)
    
    best_reward = -float('inf')
    best_prog = None
    
    iterations = 0
    while iterations < args.num_iterations:
        progs, rewards = search_method.search(search_space, task_envs)
        if rewards[-1] > best_reward:
            best_reward = rewards[-1]
            best_prog = progs[-1]
        iterations += len(progs)
        print(f'Iterations: {iterations}, Best Reward: {best_reward}')
        if best_reward == 1:
            break
    
    print(dsl.parse_node_to_str(best_prog))
    