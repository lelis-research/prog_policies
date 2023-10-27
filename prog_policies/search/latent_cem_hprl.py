from __future__ import annotations
from functools import partial

import torch

from prog_policies.base import BaseEnvironment

from hprl.pretrain.models_option_new_vae import ProgramVAE
from hprl.rl.envs import make_vec_envs
from hprl.karel_env.dsl import get_DSL_option_v2 as get_DSL
from hprl.pretrain.customargparse import CustomArgumentParser, args_to_dict
from hprl.fetch_mapping import fetch_mapping

from .latent_cem import LatentCEM

class LatentCEM_HPRL(LatentCEM):
    
    def load_latent_model(self, latent_model_cls_name: str, latent_model_args: dict, latent_model_params_path: str,
                          env_cls: type[BaseEnvironment] = None, env_args: dict = None):
        parser = CustomArgumentParser()
        parser.add_argument('-c', '--configfile')
        parser.set_defaults(configfile='hprl/pretrain/cfg_option_new_vae.py')
        args, _ = parser.parse_known_args()
        _, _, args.dsl_tokens, _ = fetch_mapping('hprl/mapping_karel2prl_new_vae_v2.txt')
        # -d datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch --verbose --train.batch_size 256 --num_lstm_cell_units 64 --net.num_rnn_encoder_units 256 --net.num_rnn_decoder_units 256 --loss.latent_loss_coef 0.1 --net.use_linear True --net.tanh_after_sample True --device cuda:0 --mdp_type ProgramEnv1_new_vae_v2 --optimizer.params.lr 1e-3 --net.latent_mean_pooling False --prefix LEAPSL_tanh_epoch30_L40_1m_h64_u256_option_latent_p1_gru_linear_cuda8 --max_program_len 40 --dsl.max_program_len 40 --input_channel 8 --train.max_epoch 30
        args.use_simplified_dsl = False
        args.device = 'cpu'
        args.num_lstm_cell_units = 64
        args.mdp_type = "ProgramEnv1_new_vae_v2"
        config = args_to_dict(args)
        args.task_file = config['rl']['envs']['executable']['task_file']
        args.grammar = config['dsl']['grammar']
        args.use_simplified_dsl = config['dsl']['use_simplified_dsl']
        args.task_definition = config['rl']['envs']['executable']['task_definition']
        args.execution_guided = config['rl']['policy']['execution_guided']
        config['args'] = args
        torch.manual_seed(latent_model_args['seed'])
        envs = make_vec_envs(config['env_name'], latent_model_args['seed'], self.n_proc,
                            config['gamma'], None, self.torch_device, False,
                            custom_env=True, custom_kwargs={"config": config['args']})
        self.leaps_dsl = get_DSL(seed=latent_model_args['seed'])
        config['dsl']['num_agent_actions'] = len(self.leaps_dsl.action_functions) + 1
        self.latent_model = ProgramVAE(envs, **config)
        params = torch.load(latent_model_params_path, map_location=self.torch_device)
        self.latent_model.load_state_dict(params[0], strict=False)
        self.hidden_size = self.latent_model.recurrent_hidden_state_size
    
    def decode_population(self, population: torch.Tensor) -> list[str]:
        _, progs, progs_len, _, _, _, _, _, _ = self.latent_model.vae.decoder(None, population, teacher_enforcing=False, deterministic=True, evaluate=False)
        progs = progs.numpy().tolist()
        progs_len = progs_len.numpy().tolist()
        progs_str = [self.leaps_dsl.intseq2str([0] + prog[:prog_len[0]]) for prog, prog_len in zip(progs, progs_len)]
        return progs_str
        