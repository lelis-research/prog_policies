from abc import ABC, abstractmethod
import numpy as np
import torch

from ..base.dsl import BaseDSL

class BaseSearchSpace(ABC):
    
    def __init__(self, dsl: BaseDSL) -> None:
        super().__init__()
        self.dsl = dsl
        self.np_rng = np.random.RandomState(1)
        self.torch_device = torch.device('cpu')
        self.torch_rng = torch.Generator(device=self.torch_device)
        self.torch_rng.manual_seed(self.np_rng.randint(1000000))
    
    def initialize_program(self, seed: int = None):
        if seed is not None:
            self.np_rng = np.random.RandomState(seed)
            self.torch_rng.manual_seed(self.np_rng.randint(1000000))
        self.current_program = self.random_program()
        
    def get_current_program(self):
        return self.current_program
    
    @abstractmethod
    def random_program(self):
        pass
    
    @abstractmethod
    def mutate_current_program(self):
        pass
    
    @abstractmethod
    def rollback_mutation(self):
        pass
