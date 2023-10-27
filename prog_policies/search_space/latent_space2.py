from __future__ import annotations
import torch

from ..base.dsl import dsl_nodes, BaseDSL
from leaps.pretrain.models import ProgramVAE
from leaps.rl.envs import make_vec_envs
from leaps.karel_env.dsl import get_DSL
from leaps.pretrain.customargparse import CustomArgumentParser, args_to_dict
from leaps.fetch_mapping import fetch_mapping

from .base_space import BaseSearchSpace

class LatentSpace2(BaseSearchSpace):
    
    def __init__(self, dsl: BaseDSL, sigma: float = 0.25) -> None:
        super().__init__(dsl)
        torch.set_num_threads(1)
        parser = CustomArgumentParser()
        parser.add_argument('-c', '--configfile')
        parser.set_defaults(configfile='leaps/pretrain/cfg.py')
        args, _ = parser.parse_known_args()
        _, _, args.dsl_tokens, _ = fetch_mapping('leaps/mapping_karel2prl.txt')
        args.use_simplified_dsl = False
        args.device = 'cpu'
        args.num_lstm_cell_units = 256
        config = args_to_dict(args)
        args.task_file = config['rl']['envs']['executable']['task_file']
        args.grammar = config['dsl']['grammar']
        args.use_simplified_dsl = config['dsl']['use_simplified_dsl']
        args.task_definition = config['rl']['envs']['executable']['task_definition']
        args.execution_guided = config['rl']['policy']['execution_guided']
        config['args'] = args
        seed = self.torch_rng.get_state().cpu().numpy()[0]
        torch.manual_seed(seed)
        envs = make_vec_envs(config['env_name'], seed, 1,
                            config['gamma'], None, self.torch_device, False,
                            custom_env=True, custom_kwargs={'config': config['args']})
        self.leaps_dsl = get_DSL(seed=seed)
        config['dsl']['num_agent_actions'] = len(self.leaps_dsl.action_functions) + 1
        self.latent_model = ProgramVAE(envs, **config)
        params = torch.load('leaps/weights/LEAPS/best_valid_params.ptp', map_location=self.torch_device)
        self.latent_model.load_state_dict(params[0], strict=False)
        self.hidden_size = self.latent_model.recurrent_hidden_state_size
        self.sigma = sigma
    
    def decode_individual(self, individual: torch.Tensor) -> dsl_nodes.Program:
        population = individual.unsqueeze(0)
        _, progs, progs_len, _, _, _, _, _, _ = self.latent_model.vae.decoder(None, population, teacher_enforcing=False, deterministic=True, evaluate=False)
        prog = progs.numpy().tolist()[0]
        prog_len = progs_len.numpy().tolist()[0][0]
        prog_str = self.leaps_dsl.intseq2str([0] + prog[:prog_len])
        prog = self.dsl.parse_str_to_node(prog_str)
        return prog
    
    def initialize_individual(self) -> tuple[torch.Tensor, dsl_nodes.Program]:
        while True:
            try:
                latent = torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
                prog = self.decode_individual(latent) # Check if it's a valid program
                break
            except (AssertionError, IndexError): # In case of invalid program, try again
                continue
        return latent, prog
    
    def get_neighbors(self, individual: torch.Tensor, k: int = 1) -> list[tuple[torch.Tensor, dsl_nodes.Program]]:
        neighbors = []
        init_prog = self.decode_individual(individual)
        n_nodes = len(init_prog.get_all_nodes())
        for _ in range(k):
            n_tries = 0
            while n_tries < 50:
                try:
                    if self.np_rng.rand() < 1/n_nodes:
                        neighbor = torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
                    else:
                        neighbor = individual + self.sigma * torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
                    prog = self.decode_individual(neighbor) # Check if it's a valid program
                    break
                except (AssertionError, IndexError): # In case of invalid program, try again
                    n_tries += 1
                    continue
            if n_tries == 50: raise Exception("Couldn't find a valid mutation")
            neighbors.append((neighbor, prog))
        return neighbors
