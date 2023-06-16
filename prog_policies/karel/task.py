import numpy as np
from typing import Union

from prog_policies.base import BaseTask

class KarelTask(BaseTask):
    
    def __init__(self, seed: Union[int, None] = None, env_args: dict = {}):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.env_args = env_args
        self.crash_penalty = -100.
        super().__init__()
