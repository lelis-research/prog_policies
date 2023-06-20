import sys

sys.path.append('.')

from prog_policies.karel import KarelDSL
from prog_policies.karel_tasks import get_task_cls
from prog_policies.search.simulated_annealing import SimulatedAnnealing
from prog_policies.output_handler import OutputHandler
from prog_policies.args_handler import parse_args

if __name__ == '__main__':
    
    args = parse_args()
    
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
    
    search_args = {
        'dsl': dsl,
        'task_cls': task_cls,
        'env_args': env_args,
        'number_executions': args.search_number_executions,
        'number_iterations': args.search_number_iterations,
        'sigma': 0.5,
        'alpha': 0.9,
        'beta': 200,
        'seed': args.search_seed,
        'output_handler': output,
    }
    searcher = SimulatedAnnealing(**search_args)
    
    filled_program, num_eval, converged = searcher.search()
