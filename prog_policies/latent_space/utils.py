import torch
import torch.nn as nn
import numpy as np

from prog_policies.base import BaseEnvironment

class EnvironmentBatch:
    
    def __init__(self, states: np.ndarray, params: dict[str, float]):
        self.envs: list[BaseEnvironment] = []
        for s in states:
            self.envs.append(BaseEnvironment(initial_state=s, **params))

    def step(self, actions):
        assert len(self.envs) == len(actions)
        for env, action_index in zip(self.envs, actions):
            if action_index < 5: # Action 5 is the "do nothing" action, for filling up empty space in the array
                env.run_action_index(action_index)
        return np.array([env.get_state() for env in self.envs])


def init_gru(module: torch.nn.GRU):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)


def init(module, weight_init, bias_init, gain=1.):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module