from __future__ import annotations
from functools import partial

import torch
import numpy as np

from prog_policies.base import dsl_nodes, BaseEnvironment

from leaps.pretrain.models import ProgramVAE
from leaps.rl.envs import make_vec_envs
from leaps.karel_env.dsl import get_DSL
from leaps.pretrain.customargparse import CustomArgumentParser, args_to_dict
from leaps.fetch_mapping import fetch_mapping

from .latent_cem_leaps import LatentCEM_LEAPS

class LatentCEM_LEAPS_ProgInit(LatentCEM_LEAPS):
    
    def fill_children_of_node(self, node: dsl_nodes.BaseNode,
                          current_depth: int = 1, current_sequence: int = 0,
                          max_depth: int = 4, max_sequence: int = 6) -> None:
        node_prod_rules = self.dsl.prod_rules[type(node)]
        for i, child_type in enumerate(node.get_children_types()):
            child_probs = self.dsl.get_dsl_nodes_probs(child_type)
            for child_type in child_probs:
                if child_type not in node_prod_rules[i]:
                    child_probs[child_type] = 0.
                if current_depth >= max_depth and child_type.get_node_depth() > 0:
                    child_probs[child_type] = 0.
            if issubclass(type(node), dsl_nodes.Concatenate) and current_sequence + 1 >= max_sequence:
                if dsl_nodes.Concatenate in child_probs:
                    child_probs[dsl_nodes.Concatenate] = 0.
            
            p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
            child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
            child_instance = child()
            if child.get_number_children() > 0:
                if issubclass(type(node), dsl_nodes.Concatenate):
                    self.fill_children_of_node(child_instance, current_depth + child.get_node_depth(),
                                               current_sequence + 1, max_depth, max_sequence)
                else:
                    self.fill_children_of_node(child_instance, current_depth + child.get_node_depth(),
                                               1, max_depth, max_sequence)
            
            elif isinstance(child_instance, dsl_nodes.Action):
                child_instance.name = self.np_rng.choice(list(self.dsl.action_probs.keys()),
                                                         p=list(self.dsl.action_probs.values()))
            elif isinstance(child_instance, dsl_nodes.BoolFeature):
                child_instance.name = self.np_rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                         p=list(self.dsl.bool_feat_probs.values()))
            elif isinstance(child_instance, dsl_nodes.ConstInt):
                child_instance.value = self.np_rng.choice(list(self.dsl.const_int_probs.keys()),
                                                          p=list(self.dsl.const_int_probs.values()))
            node.children[i] = child_instance
            child_instance.parent = node
    
    def random_program(self) -> dsl_nodes.Program:
        program = dsl_nodes.Program()
        self.fill_children_of_node(program, max_depth=4, max_sequence=6)
        return program
    
    def init_population(self) -> torch.Tensor:
        init_prog = self.random_program()
        init_prog_str = self.dsl.parse_node_to_str(init_prog)
        init_prog_intseq = self.leaps_dsl.str2intseq(init_prog_str)
        init_prog_len = torch.tensor([len(init_prog_intseq)])
        init_prog_intseq += [self.dsl.t2i['<pad>']] * (45 - init_prog_len)
        init_prog_torch = torch.tensor(init_prog_intseq, dtype=torch.long, device=self.torch_device).unsqueeze(0)
        _, init_enc_hxs = self.latent_model.vae.encoder(init_prog_torch, init_prog_len)
        init_latent = self.latent_model.vae._sample_latent(init_enc_hxs.squeeze(0))
        init_latent = init_latent.squeeze(0)
        population = []
        for _ in range(self.population_size):
            population.append(
                init_latent + self.sigma * torch.randn(self.hidden_size,
                                                       generator=self.torch_rng,
                                                       device=self.torch_device)
            )
        return torch.stack(population)
        