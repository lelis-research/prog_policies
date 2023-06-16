from abc import ABC, abstractmethod
import copy

from .environment import BaseEnvironment
from . import dsl_nodes

class BaseTask(ABC):
    
    def __init__(self):
        self.initial_environment = self.generate_initial_environment()
        self.reset_environment()
    
    def get_environment(self) -> BaseEnvironment:
        return self.environment
    
    def reset_environment(self) -> None:
        self.environment = copy.deepcopy(self.initial_environment)
    
    @abstractmethod
    def generate_initial_environment(self) -> BaseEnvironment:
        pass
    
    @abstractmethod
    def get_reward(self, environment: BaseEnvironment) -> tuple[bool, float]:
        pass
    
    def evaluate_program(self, program: dsl_nodes.Program) -> float:
        self.reset_environment()
        reward = 0.
        for _ in program.run_generator(self.environment):
            terminated, instant_reward = self.get_reward(self.environment)
            reward += instant_reward
            if terminated or self.environment.is_crashed():
                break
        return reward

    def trace_program(self, program: dsl_nodes.Program, image_name = 'trace.gif', max_steps = 50):
        from PIL import Image
        self.reset_environment()
        im = Image.fromarray(self.to_image())
        im_list = []
        for _ in program.run_generator(self.environment):
            terminated, _ = self.get_reward(self.environment)
            im_list.append(Image.fromarray(self.to_image()))
            if len(im_list) > max_steps or terminated or self.is_crashed():
                break
        im.save(image_name, save_all=True, append_images=im_list, duration=75, loop=0)
