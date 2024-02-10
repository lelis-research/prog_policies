from __future__ import annotations
import copy
import sys
import time
import numpy as np
from tqdm import tqdm

sys.path.append('.')

from prog_policies.base import dsl_nodes
from prog_policies.karel import KarelDSL, KarelEnvironment, KarelStateGenerator
from prog_policies.search_space import BaseSearchSpace, ProgrammaticSpace, LatentSpace, LatentSpace2



if __name__ == '__main__':
    
    n_passes = 1000
    
    dsl = KarelDSL()
    search_spaces: list[BaseSearchSpace] = [
        ProgrammaticSpace(dsl, 0.25),
        LatentSpace(dsl, sigma=0.25),
    ]
    search_spaces_labels = [
        'programmatic',
        'latent',
    ]
    
    env_args = {
        "env_height": 8,
        "env_width": 8,
        "crashable": False,
        "leaps_behaviour": True,
        "max_calls": 100
    }
    
    env_generator = KarelStateGenerator(env_args, 0)
    
    for space, label in zip(search_spaces, search_spaces_labels):
        running_times = []
        for i in range(n_passes):
            space.set_seed(i)
            ind, _ = space.initialize_individual()
            t = time.time()
            _ = space.get_neighbors(ind, k=1)
            running_times.append(time.time() - t)
        with open(f'output/{label}_running_times.csv', 'w') as f:
            f.write(','.join([str(r) for r in running_times]) + '\n')
        running_times_err = 1.96 * np.std(running_times) / np.sqrt(n_passes)
        print(f'{label} mean running time: {np.mean(running_times)} +- {running_times_err}')
        print(f'std: {np.std(running_times)}')
        
