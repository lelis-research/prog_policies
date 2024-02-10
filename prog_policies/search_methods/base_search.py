from __future__ import annotations
from abc import ABC, abstractmethod

from ..search_space import BaseSearchSpace
from ..base import dsl_nodes, BaseTask


class BaseSearch(ABC):
    
    def __init__(self, k: int, e: int = None) -> None:
        self.k = k
        self.e = e
    
    @abstractmethod
    def search(self, search_space: BaseSearchSpace, task_envs: list[BaseTask],
               seed = None, n_iterations: int = 10000) -> tuple[list[dsl_nodes.Program], list[float]]:
        """Main method for searching in a given search space

        Args:
            search_space (BaseSearchSpace): Search space instance
            task_envs (list[BaseTask]): List of task environments for evaluation
            seed (int, optional): If provided, sets the search space RNG seed. Defaults to None.
            n_iterations (int, optional): Maximum number of iterations. Defaults to 10000.

        Returns:
            list[dsl_nodes.Program]: List of programs obtained at each iteration
            list[float]: List of rewards obtained at each iteration
        """
        pass
    
    def evaluate_program(self, program: dsl_nodes.Program, task_envs: list[BaseTask]) -> float:
        """Evaluates a program in a list of task environments

        Args:
            program (dsl_nodes.Program): Input program
            task_envs (list[BaseTask]): List of task environments

        Returns:
            float: Mean episodic return obtained in the task environments
        """
        sum_reward = 0.
        for task_env in task_envs:
            sum_reward += task_env.evaluate_program(program)
        return sum_reward / len(task_envs)
