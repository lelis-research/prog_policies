from __future__ import annotations
import os
import pickle
import sys
import numpy as np
import tqdm

sys.path.append('.')

from prog_policies.karel import KarelDSL, KarelProgramGenerator, KarelStateGenerator

if __name__ == '__main__':
    
    dsl = KarelDSL()

    env_args = {
        "env_height": 8,
        "env_width": 8,
        "crashable": False,
        "leaps_behaviour": True,
        "max_calls": 10000
    }
    
    num_programs = 100000
    max_program_size = 32
    max_program_depth = 4
    max_program_seq = 8
    
    num_demos = 32
    max_demo_length = 256
    
    dataset_identifier = f'{num_programs}progs_{max_program_size}size_{max_program_depth}depth_{max_program_seq}seq_{num_demos}demos_{max_demo_length}demo_len'
    
    state_generator = KarelStateGenerator(env_args)
    prog_generator = KarelProgramGenerator(dsl)
    
    seen_programs = set()
    programs_only_dataset = []
    full_programs_dataset = []
    
    with tqdm.tqdm(total=num_programs) as pbar:

        while len(programs_only_dataset) < num_programs:
            program = prog_generator.generate_program(max_program_depth, max_program_size,
                                                      max_program_seq)
            
            program_str = dsl.parse_node_to_str(program)
            if program_str in seen_programs:
                continue
            
            try:
                s_h, bf_h, a_h = prog_generator.generate_demos(program, state_generator,
                                                                num_demos, max_demo_length)
            except Exception:
                continue
            
            seen_programs.add(program_str)
            
            # max_len = max([a.size for a in a_h])
            
            # a_h_np = np.ones((num_demos, max_len+1), dtype=int) * (prog_generator.a2i[None])
            # s_h_np = np.zeros((num_demos, max_len+1, *s_h[0].shape[1:]), dtype=bool)
            # bf_h_np = np.zeros((num_demos, max_len+1, *bf_h[0].shape[1:]), dtype=bool)
            
            # for i in range(num_demos):
            #     a_h_np[i, 1:len(a_h[i])+1] = a_h[i]
            #     s_h_np[i, :len(a_h[i])+1] = s_h[i]
            #     s_h_np[i, len(a_h[i])+1:] = s_h[i][-1]
            #     bf_h_np[i, :len(a_h[i])+1] = bf_h[i]
            #     bf_h_np[i, len(a_h[i])+1:] = bf_h[i][-1]
            
            programs_only_dataset.append(program_str)
            full_programs_dataset.append((program_str, (s_h, bf_h, a_h)))
            
            pbar.update(1)
    
    with open(f'data/{dataset_identifier}.pkl', 'wb') as f:
        pickle.dump(full_programs_dataset, f)
    