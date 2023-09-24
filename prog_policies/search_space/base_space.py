from __future__ import annotations
from typing import Any
from abc import ABC, abstractmethod
import numpy as np
import torch

from ..base.dsl import BaseDSL, dsl_nodes

class BaseSearchSpace(ABC):
    
    def __init__(self, dsl: BaseDSL) -> None:
        super().__init__()
        self.dsl = dsl
        self.np_rng = np.random.RandomState(1)
        self.torch_device = torch.device('cpu')
        self.torch_rng = torch.Generator(device=self.torch_device)
        self.torch_rng.manual_seed(self.np_rng.randint(1000000))

    def set_seed(self, seed: int):
        self.np_rng = np.random.RandomState(seed)
        self.torch_rng.manual_seed(self.np_rng.randint(1000000))
    
    @abstractmethod
    def initialize_individual(self, seed: int = None) -> tuple[Any, dsl_nodes.Program]:
        pass
    
    @abstractmethod
    def get_neighbors(self, individual, k: int = 1) -> list[tuple[Any, dsl_nodes.Program]]:
        pass
