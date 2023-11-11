from __future__ import annotations
import torch

from ..base.dsl import dsl_nodes

from .latent_space import LatentSpace

class LatentSpace2(LatentSpace):

    def get_neighbors(self, individual: torch.Tensor, k: int = 1) -> list[tuple[torch.Tensor, dsl_nodes.Program]]:
        neighbors = []
        init_prog = self._decode(individual)
        n_nodes = len(init_prog.get_all_nodes())
        for _ in range(k):
            n_tries = 0
            while n_tries < 50:
                try:
                    if self.np_rng.rand() < 1/n_nodes:
                        neighbor = torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
                    else:
                        neighbor = individual + self.sigma * torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
                    prog = self._decode(neighbor) # Check if it's a valid program
                    break
                except (AssertionError, IndexError): # In case of invalid program, try again
                    n_tries += 1
                    continue
            if n_tries == 50: raise Exception("Couldn't find a valid mutation")
            neighbors.append((neighbor, prog))
        return neighbors
