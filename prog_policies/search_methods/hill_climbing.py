from __future__ import annotations

from ..search_space import BaseSearchSpace
from ..base import dsl_nodes, BaseTask

from .base_search import BaseSearch


class HillClimbing(BaseSearch):

    def search(self, search_space: BaseSearchSpace, task_envs: list[BaseTask],
               seed: int | None = None, n_iterations: int = 10000, init = None) -> tuple[list[dsl_nodes.Program], list[float]]:
        """Performs hill climbing in the search space (any search space can be used), stopping when
        a local maximum is reached or when the maximum number of iterations is reached

        Args:
            search_space (BaseSearchSpace): Search space instance
            task_envs (list[BaseTask]): List of task environments for evaluation
            seed (int, optional): If provided, sets the search space RNG seed. Defaults to None.
            n_iterations (int, optional): Maximum number of iterations. Defaults to 10000.

        Returns:
            list[float]: List of rewards obtained at each iteration
        """
        rewards = []
        if seed:
            search_space.set_seed(seed)
        if init is None:
            best_ind, best_prog = search_space.initialize_individual()
        else:
            best_ind, best_prog = init
        best_reward = self.evaluate_program(best_prog, task_envs)
        rewards.append(best_reward)
        progs = [best_prog]
        for _ in range(n_iterations):
            candidates = search_space.get_neighbors(best_ind, k=self.k)
            in_local_maximum = True
            for ind, prog in candidates:
                reward = self.evaluate_program(prog, task_envs)
                if reward > best_reward:
                    best_ind = ind
                    best_prog = prog
                    best_reward = reward
                    in_local_maximum = False
                    break
            if in_local_maximum:
                break
            rewards.append(best_reward)
            progs.append(best_prog)
        return progs, rewards
