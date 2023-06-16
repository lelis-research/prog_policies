import torch

from .base_vae import ModelOutput
from .leaps_vae import LeapsVAE

class SketchVAE(LeapsVAE):
    
    def forward(self, s_h: torch.Tensor, a_h: torch.Tensor, a_h_mask: torch.Tensor, 
                prog: torch.Tensor, prog_mask: torch.Tensor, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelOutput:
        z = self.encode(prog, prog_mask)
        
        decoder_result = self.decode(z, prog, prog_mask, prog_teacher_enforcing)
        pred_progs, pred_progs_logits, pred_progs_masks = decoder_result
        
        return ModelOutput(pred_progs, pred_progs_logits, pred_progs_masks,
                           None, None, None)
