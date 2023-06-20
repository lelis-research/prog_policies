import sys
import torch

sys.path.append('.')

from prog_policies.karel import KarelDSL, KarelEnvironment
from prog_policies.karel_tasks import get_task_cls
from prog_policies.latent_space.models import *
from prog_policies.search.latent_cem import LatentCEM
from prog_policies.output_handler import OutputHandler
from prog_policies.args_handler import parse_args

if __name__ == '__main__':
    
    args = parse_args()
    
    device = torch.device('cpu')
    
    dsl = KarelDSL()
    
    output = OutputHandler(
        experiment_name=args.experiment_name,
        log_filename=args.log_filename
    )
    
    task_cls = get_task_cls(args.env_task)
    
    env_args = {
        'env_height': args.env_height,
        'env_width': args.env_width,
        'crashable': args.env_is_crashable,
        'leaps_behaviour': args.env_enable_leaps_behaviour,
        'max_calls': 10000
    }
    env = KarelEnvironment(**env_args)
    
    model_cls = globals()[args.model_name]
    
    model_args = {
        'hidden_size': args.model_hidden_size,
        'max_demo_length': args.data_max_demo_length,
        'max_program_length': args.data_max_program_length,
    }
    
    model = model_cls(dsl, env, device, **model_args)
    model_params = torch.load(args.model_params_path, map_location=device)
    model.load_state_dict(model_params)
    
    search_args = {
        'population_size': args.search_population_size,
        'elitism_rate': args.search_elitism_rate,
        'number_executions': args.search_number_executions,
        'number_iterations': args.search_number_iterations,
        'reduce_to_mean': args.search_reduce_to_mean,
        'sigma': args.search_sigma,
        'restart_timeout': args.search_restart_timeout,
        'output_handler': output,
        'multiprocessing': args.multiprocessing_active
    }
    searcher = LatentCEM(model, task_cls, dsl, env_args, **search_args)
    
    best_program, num_eval, converged = searcher.search()
