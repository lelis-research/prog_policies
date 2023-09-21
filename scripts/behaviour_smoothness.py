from __future__ import annotations
import copy
import sys
import numpy as np
from tqdm import tqdm

sys.path.append('.')

from prog_policies.base import dsl_nodes
from prog_policies.karel import KarelDSL, KarelEnvironment, KarelStateGenerator
from prog_policies.search_space import BaseSearchSpace, ProgrammaticSpace, LatentSpace


def get_trajectory(program: dsl_nodes.Program, initial_state: KarelEnvironment) -> list[dsl_nodes.Action]:
    tau = []
    for action in program.run_generator(copy.deepcopy(initial_state)):
        tau.append(action)
    return tau

def trajectories_similarity(tau_a: list[dsl_nodes.Action], tau_b: list[dsl_nodes.Action]) -> float:
    similarity = 0.
    t_max = max(len(tau_a), len(tau_b))
    if t_max == 0: return 1.
    for i in range(min(len(tau_a), len(tau_b))):
        if tau_a[i].name == tau_b[i].name:
            similarity += 1.
        else:
            break
    similarity /= t_max
    return similarity

def behaviour_smoothness_one_pass(search_space: BaseSearchSpace, env_generators: list[KarelStateGenerator],
                                  n_mutations: int = 10) -> tuple[list[float], list[str]]:
    search_space.initialize_program()
    initial_states = [env_generator.random_state() for env_generator in env_generators]
    initial_program = search_space.get_current_program()
    initial_trajectories = [get_trajectory(initial_program, s) for s in initial_states]
    smoothness = []
    programs = [search_space.dsl.parse_node_to_str(initial_program)]
    for _ in range(n_mutations):
        search_space.mutate_current_program()
        current_program = search_space.get_current_program()
        trajectories = [get_trajectory(current_program, s) for s in initial_states]
        smoothness.append(np.mean([trajectories_similarity(tau_a, tau_b) for tau_a, tau_b in zip(initial_trajectories, trajectories)]))
        programs.append(search_space.dsl.parse_node_to_str(current_program))
    return smoothness, programs

if __name__ == '__main__':
    
    n_passes = 1000
    n_env = 32
    n_mutations = 10
    
    dsl = KarelDSL()
    search_spaces = [
        # ProgrammaticSpace(dsl),
        LatentSpace(dsl)
    ]
    search_spaces_labels = [
        # 'programmatic',
        'latent'
    ]
    
    env_args = {
        "env_height": 8,
        "env_width": 8,
        "crashable": False,
        "leaps_behaviour": True,
        "max_calls": 100
    }
    
    env_generators = [KarelStateGenerator(env_args, i) for i in range(n_env)]
    
    for space, label in zip(search_spaces, search_spaces_labels):
        smoothness_list, programs_list = [], []
        while len(smoothness_list) < n_passes:
            try:
                smoothness, programs = behaviour_smoothness_one_pass(space, env_generators, n_mutations)
                smoothness_list.append(smoothness)
                programs_list.append(programs)
            except Exception:
                continue
        with open(f'output/{label}_smoothness_values.csv', 'w') as f:
            f.write(','.join([str(s) for s in range(1, n_mutations+1)]) + '\n')
            for smoothness in smoothness_list:
                f.write(','.join([str(s) for s in smoothness]) + '\n')
        with open(f'output/{label}_programs.csv', 'w') as f:
            for programs in programs_list:
                f.write(','.join(programs) + '\n')
