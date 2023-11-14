from __future__ import annotations

import torch

from ..search_space import LatentSpace
from ..base import BaseTask

from .base_search import BaseSearch


class CEBS(BaseSearch):
    
    def search(self, search_space: LatentSpace, task_envs: list[BaseTask],
               seed = None, n_iterations: int = 10000) -> list[float]:
        rewards = []
        search_space.set_seed(seed)
        best_ind, best_prog = search_space.initialize_individual()
        best_reward = self.evaluate_program(best_prog, task_envs)
        rewards.append(best_reward)
        best_elite_mean = -float('inf')

        candidates = search_space.get_neighbors(best_ind, k=self.k)        
        for _ in range(n_iterations):
            
            candidate_rewards = []
            for _, prog in candidates:
                reward = self.evaluate_program(prog, task_envs)
                candidate_rewards.append(reward)
                if reward > best_reward:
                    best_reward = reward
            rewards.append(best_reward)
            
            torch_candidates = torch.stack([ind for ind, _ in candidates])
            torch_rewards = torch.tensor(candidate_rewards, device=torch_candidates.device)
            
            elite_indices = torch.topk(torch_rewards, self.e, largest=True).indices
            elite_candidates = torch_candidates[elite_indices]
            elite_rewards = torch_rewards[elite_indices]
            
            mean_elite_reward = torch.mean(elite_rewards, dim=0)
            if mean_elite_reward > best_elite_mean:
                best_elite_mean = mean_elite_reward
            else:
                break
            
            candidates = []
            for candidate in elite_candidates:
                candidates += search_space.get_neighbors(candidate, k=self.k//self.e)
            
        return rewards