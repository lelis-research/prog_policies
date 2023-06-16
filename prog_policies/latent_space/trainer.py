import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import NamedTuple

from prog_policies.output_handler import OutputHandler

from .models.base_vae import BaseVAE


class EpochReturn(NamedTuple):
    mean_total_loss: float
    mean_progs_loss: float
    mean_a_h_loss: float
    mean_latent_loss: float
    mean_progs_t_accuracy: float
    mean_progs_s_accuracy: float
    mean_a_h_t_accuracy: float
    mean_a_h_s_accuracy: float


class Trainer:
    
    def __init__(self, model: BaseVAE, prog_loss_coef: float = 1.0,
                 a_h_loss_coef: float = 1.0, latent_loss_coef: float = 0.1,
                 disable_prog_teacher_enforcing: bool = False,
                 disable_a_h_teacher_enforcing: bool = False,
                 optim_lr: float = 5e-4, save_params_each_epoch: bool = False,
                 num_epochs: int = 100, output_handler: OutputHandler = None):
        self.model = model
        self.prog_loss_coef = prog_loss_coef
        self.a_h_loss_coef = a_h_loss_coef
        self.latent_loss_coef = latent_loss_coef
        self.disable_prog_teacher_enforcing = disable_prog_teacher_enforcing
        self.disable_a_h_teacher_enforcing = disable_a_h_teacher_enforcing
        self.save_each_epoch = save_params_each_epoch
        self.num_epochs = num_epochs
        self.output_handler = output_handler
        self.device = self.model.device
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=optim_lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=.95
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
        if self.output_handler is not None:
            self.output_handler.setup_trainer()
    
    def _log(self, message: str):
        if self.output_handler is not None:
            self.output_handler.log('Trainer', message)
    
    def _run_batch(self, batch: list, training = True) -> list:
        if training:
            self.model.train()
            torch.set_grad_enabled(True) # prob not needed
        else:
            self.model.eval()
            torch.set_grad_enabled(False) # prob not needed
            
        s_h, a_h, a_h_masks, progs, progs_masks = batch
        
        output = self.model(s_h, a_h, a_h_masks, progs, progs_masks,
                            not self.disable_prog_teacher_enforcing,
                            not self.disable_a_h_teacher_enforcing)
        pred_progs, pred_progs_logits, pred_progs_masks,\
            pred_a_h, pred_a_h_logits, pred_a_h_masks = output
        
        if type(a_h) == torch.Tensor:
            # Combine first 2 dimensions of a_h (batch_size and demos_per_program)
            a_h = a_h.view(-1, a_h.shape[-1])
            a_h_masks = a_h_masks.view(-1, a_h_masks.shape[-1])
            
            # Skip first token in ground truth sequences
            a_h = a_h[:, 1:].contiguous()
            a_h_masks = a_h_masks[:, 1:].contiguous()
            
            # Flatten everything for loss calculation
            a_h_flat = a_h.view(-1, 1)
            a_h_masks_flat = a_h_masks.view(-1, 1)
        
        if type(progs) == torch.Tensor:
            # Skip first token in ground truth sequences
            progs = progs[:, 1:].contiguous()
            progs_masks = progs_masks[:, 1:].contiguous()

            # Flatten everything for loss calculation
            progs_flat = progs.view(-1, 1)
            progs_masks_flat = progs_masks.view(-1, 1)
        
        if pred_progs is not None:
            pred_progs_logits = pred_progs_logits.view(-1, pred_progs_logits.shape[-1])
            pred_progs_masks_flat = pred_progs_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            progs_masks_flat_combined = torch.max(progs_masks_flat, pred_progs_masks_flat).squeeze()

        if pred_a_h is not None:
            pred_a_h_logits = pred_a_h_logits.view(-1, pred_a_h_logits.shape[-1])
            pred_a_h_masks_flat = pred_a_h_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            a_h_masks_flat_combined = torch.max(a_h_masks_flat, pred_a_h_masks_flat).squeeze()
        
        if training:
            self.optimizer.zero_grad()
        
        # Calculate classification loss only on tokens in mask
        zero_tensor = torch.tensor([0.0], device=self.device, requires_grad=False)
        progs_loss, a_h_loss = zero_tensor, zero_tensor
        
        if pred_progs is not None:
            progs_loss = self.loss_fn(pred_progs_logits[progs_masks_flat_combined],
                                    progs_flat[progs_masks_flat_combined].view(-1))

        if pred_a_h is not None:
            a_h_loss = self.loss_fn(pred_a_h_logits[a_h_masks_flat_combined],
                                    a_h_flat[a_h_masks_flat_combined].view(-1))
        
        latent_loss = self.model.get_latent_loss()
        
        total_loss = self.prog_loss_coef * progs_loss + self.a_h_loss_coef * a_h_loss \
            + self.latent_loss_coef * latent_loss
            
        if training:
            total_loss.backward()
            self.optimizer.step()
            
        with torch.no_grad():
            progs_s_accuracy, progs_t_accuracy = zero_tensor, zero_tensor
            if pred_progs is not None:
                progs_masks_combined = torch.max(progs_masks, pred_progs_masks)
                progs_t_accuracy = (pred_progs[progs_masks_combined] == progs[progs_masks_combined]).float().mean()
                progs_s_accuracy = (progs == pred_progs).min(dim=1).values.float().mean()
            
            a_h_s_accuracy, a_h_t_accuracy = zero_tensor, zero_tensor
            if pred_a_h is not None:
                a_h_masks_combined = torch.max(a_h_masks, pred_a_h_masks)
                a_h_t_accuracy = (pred_a_h[a_h_masks_combined] == a_h[a_h_masks_combined]).float().mean()
                a_h_s_accuracy = (a_h == pred_a_h).min(dim=1).values.float().mean()
            
        return [
            total_loss.detach().cpu().numpy().item(),
            progs_loss.detach().cpu().numpy().item(),
            a_h_loss.detach().cpu().numpy().item(),
            latent_loss.detach().cpu().numpy().item(),
            progs_t_accuracy.detach().cpu().numpy().item(),
            progs_s_accuracy.detach().cpu().numpy().item(),
            a_h_t_accuracy.detach().cpu().numpy().item(),
            a_h_s_accuracy.detach().cpu().numpy().item()
        ]

    def _run_epoch(self, dataloader: DataLoader, epoch: int, training = True) -> EpochReturn:
        batch_info_list = np.zeros((len(dataloader), 8))
        
        for batch_idx, batch in enumerate(dataloader):
            batch_info = self._run_batch(batch, training)
            batch_info_list[batch_idx] = batch_info
        
        epoch_info_list = np.mean(batch_info_list, axis=0)
        
        return EpochReturn(*epoch_info_list.tolist())
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        if val_dataloader is not None:
            validation_key = 'mean_total_loss'
            best_val_return = np.inf
        
        if self.output_handler is not None:
            self.output_handler.setup_training_info(EpochReturn._fields)

        if val_dataloader is not None:
            if self.output_handler is not None:
                self.output_handler.setup_validation_info(EpochReturn._fields)
        
        for epoch in range(1, self.num_epochs + 1):
            self._log(f'Training epoch {epoch}.')
            train_info = self._run_epoch(train_dataloader, epoch, True)
            self._log(train_info._asdict())
            if self.output_handler is not None:
                self.output_handler.save_training_info(epoch, train_info)
            if self.save_each_epoch:
                if self.output_handler is not None:
                    self.output_handler.save_model_parameters(f'epoch_{epoch}', self.model)
                self._log(f'Parameters saved as epoch_{epoch}.ptp')
 
            if val_dataloader is not None:
                self._log(f'Validation epoch {epoch}.')
                val_info = self._run_epoch(val_dataloader, epoch, False)
                self._log(val_info._asdict())
                if self.output_handler is not None:
                    self.output_handler.save_validation_info(epoch, val_info)
                val_return = val_info._asdict()[validation_key]
 
                if val_return < best_val_return:
                    best_val_return = val_return
                    self._log(f'New best validation {validation_key}: {best_val_return}')
                    if self.output_handler is not None:
                        self.output_handler.save_model_parameters('best_val', self.model)
                    self._log('Parameters saved as best_val.ptp')
