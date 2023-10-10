from __future__ import annotations
import copy
import sys
import numpy as np
from tqdm import tqdm

sys.path.append('.')

from prog_policies.base import dsl_nodes
from prog_policies.karel import KarelDSL, KarelEnvironment, KarelStateGenerator
from prog_policies.search_space import BaseSearchSpace, ProgrammaticSpace, LatentSpace, LatentSpace2


def get_state_sequence(program: dsl_nodes.Program, initial_state: KarelEnvironment) -> list[KarelEnvironment]:
    state = copy.deepcopy(initial_state)
    states = [state]
    for _ in program.run_generator(state):
        states.append(copy.deepcopy(state))
    return states

# normalized levenshtein distance
def lev_distance(tau_a: list[KarelEnvironment], tau_b: list[KarelEnvironment]) -> float:
    distances = np.zeros((len(tau_a) + 1, len(tau_b) + 1))
    for i in range(len(tau_a) + 1):
        distances[i, 0] = i
    for j in range(len(tau_b) + 1):
        distances[0, j] = j
    for i in range(1, len(tau_a) + 1):
        for j in range(1, len(tau_b) + 1):
            if tau_a[i-1] == tau_b[j-1]:
                distances[i, j] = distances[i-1, j-1]
            else:
                distances[i, j] = min(distances[i-1, j], distances[i, j-1], distances[i-1, j-1]) + 1
    return 1. - distances[-1, -1] / max(len(tau_a), len(tau_b))

def behaviour_smoothness_one_pass(search_space: BaseSearchSpace, env_generators: list[KarelStateGenerator],
                                  n_mutations: int = 10) -> tuple[list[float], list[str]]:
    individual, init_prog = search_space.initialize_individual()
    initial_states = [env_generator.random_state() for env_generator in env_generators]
    initial_trajectories = [get_state_sequence(init_prog, s) for s in initial_states]
    smoothness = []
    programs = [search_space.dsl.parse_node_to_str(init_prog)]
    for _ in range(n_mutations):
        individual, prog = search_space.get_neighbors(individual, k=1)[0]
        trajectories = [get_state_sequence(prog, s) for s in initial_states]
        smoothness.append(np.mean([lev_distance(tau_a, tau_b) for tau_a, tau_b in zip(initial_trajectories, trajectories)]))
        programs.append(search_space.dsl.parse_node_to_str(prog))
    return smoothness, programs

if __name__ == '__main__':
    
    n_passes = 1000
    n_env = 32
    n_mutations = 10
    
    dsl = KarelDSL()
    search_spaces = [
        ProgrammaticSpace(dsl),
        LatentSpace(dsl, sigma=0.25),
        LatentSpace(dsl, sigma=0.1),
        LatentSpace(dsl, sigma=0.5),
        LatentSpace2(dsl, sigma=0.25),
        LatentSpace2(dsl, sigma=0.1),
        LatentSpace2(dsl, sigma=0.5),
    ]
    search_spaces_labels = [
        'programmatic',
        'latent_025',
        'latent_010',
        'latent_050',
        'latent2_025',
        'latent2_010',
        'latent2_050',
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
        with open(f'output/{label}_levenshtein_values.csv', 'w') as f:
            f.write(','.join([str(s) for s in range(1, n_mutations+1)]) + '\n')
            for smoothness in smoothness_list:
                f.write(','.join([str(s) for s in smoothness]) + '\n')
        with open(f'output/{label}_programs.csv', 'w') as f:
            for programs in programs_list:
                f.write(','.join(programs) + '\n')
