from __future__ import annotations
from typing import NamedTuple
from logging import Logger
import numpy as np
import torch
from torch import nn

from prog_policies.base import BaseDSL, BaseEnvironment

from ..utils import init, EnvironmentBatch
from ..syntax_checker import SyntaxChecker


class ModelReturn(NamedTuple):
    z: torch.Tensor
    progs: torch.Tensor
    progs_logits: torch.Tensor
    progs_masks: torch.Tensor
    a_h: torch.Tensor
    a_h_logits: torch.Tensor
    a_h_masks: torch.Tensor


class BaseVAE(nn.Module):
    """Base class for all program VAEs. Implements general functions used by subclasses.
    Do not directly instantiate this class.
    """    
    def __init__(self, dsl: BaseDSL, device: torch.device, env_cls: type[BaseEnvironment],
                 env_args: dict, max_program_length = 45, max_demo_length = 100, model_seed = 1,
                 hidden_size = 256, model_params_path: str = None, logger: Logger = None,
                 name: str = None):
        super().__init__()
        
        if name is None:
            name = self.__class__.__name__
        self.name = name
        
        torch.manual_seed(model_seed)

        self.env_cls = env_cls
        self.env_args = env_args

        self.device = device
        
        self.max_demo_length = max_demo_length
        self.max_program_length = max_program_length
        
        # Z
        self.hidden_size = hidden_size
        
        # A
        self.num_agent_actions = len(dsl.get_actions()) + 1 # +1 because we have a NOP action
        
        # T
        self.num_program_tokens = len(dsl.get_tokens()) # dsl includes <pad> and <HOLE> tokens
        
        self.pad_token = dsl.t2i['<pad>']
        
        self.init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                    constant_(x, 0), nn.init.calculate_gain('relu'))
        
        # CxHxW
        env = self.env_cls(**self.env_args)
        self.state_shape = env.state_shape
        
        self.perceptions_size = len(env.bool_features_list)
        
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # syntax_checker_tokens = dsl.get_tokens()
        # syntax_checker_tokens.append('<pad>')
        # self.T2I = {token: i for i, token in enumerate(syntax_checker_tokens)}
        self.syntax_checker = SyntaxChecker(dsl, self.device)
        
        self.to(self.device)
        
        if model_params_path is not None:
            self.load_state_dict(torch.load(model_params_path, map_location=self.device))
            
        self.logger = logger

    def env_init(self, states: torch.Tensor):
        states_np = states.detach().cpu().numpy().astype(np.bool_)
        # C x H x W to H x W x C
        # states_np = np.moveaxis(states_np,[-1,-2,-3], [-2,-3,-1])
        self._envs = EnvironmentBatch(states_np)

    def env_step(self, actions: torch.Tensor):
        # states_np = states.detach().cpu().numpy().astype(np.bool_)
        # C x H x W to H x W x C
        # states_np = np.moveaxis(states_np,[-1,-2,-3], [-2,-3,-1])
        # assert states_np.shape[-1] == 16
        # karel world expects H x W x C
        new_states = self._envs.step(actions.detach().cpu().numpy())
        # new_states = np.moveaxis(new_states,[-1,-2,-3], [-3,-1,-2])
        new_states = torch.tensor(new_states, dtype=torch.float32, device=self.device)
        return new_states
    
    def log(self, message: str):
        if self.logger is not None:
            self.logger.info(f'[{self.__class__.__name__}] {message}')
        else:
            print(f'[{self.__class__.__name__}] {message}')
    
    def get_syntax_mask(self, batch_size: int, current_tokens: torch.Tensor, grammar_state: list):
        out_of_syntax_list = []
        out_of_syntax_mask = torch.zeros((batch_size, self.num_program_tokens), dtype=torch.bool, device=self.device)

        for program_idx, inp_token in enumerate(current_tokens):
            inp_dsl_token = inp_token.detach().cpu().numpy().item()
            out_of_syntax_list.append(self.syntax_checker.get_sequence_mask(grammar_state[program_idx],
                                                                            [inp_dsl_token]).to(self.device))
        torch.cat(out_of_syntax_list, 0, out=out_of_syntax_mask)
        out_of_syntax_mask = out_of_syntax_mask.squeeze()
        syntax_mask = torch.where(out_of_syntax_mask,
                                  -torch.finfo(torch.float32).max * torch.ones_like(out_of_syntax_mask).float(),
                                  torch.zeros_like(out_of_syntax_mask).float())

        return syntax_mask, grammar_state

    def forward(self, data_batch: tuple, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelReturn:
        raise NotImplementedError
    
    def encode_program(self, prog: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def decode_vector(self, z: torch.Tensor) -> list[list[int]]:
        raise NotImplementedError
