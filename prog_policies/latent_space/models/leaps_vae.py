from __future__ import annotations
from logging import Logger
import os
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from prog_policies.base import BaseDSL, BaseEnvironment

from ..utils import init_gru
from .base_vae import BaseVAE, ModelReturn


class Accuracies(NamedTuple):
    progs_t: torch.Tensor
    progs_s: torch.Tensor
    a_h_t: torch.Tensor
    a_h_s: torch.Tensor


class Losses(NamedTuple):
    progs_rec: torch.Tensor
    a_h_rec: torch.Tensor
    latent: torch.Tensor


class LeapsVAE(BaseVAE):
    """Reproduction of program VAE used in LEAPS paper.
    """    
    def __init__(self, dsl: BaseDSL, device: torch.device, env_cls: type[BaseEnvironment],
                 env_args: dict, max_program_length = 45, max_demo_length = 100, model_seed = 1,
                 hidden_size = 256, logger: Logger = None, name: str = None):
        super().__init__(dsl, device, env_cls, env_args, max_program_length, max_demo_length,
                         model_seed, hidden_size, logger=logger, name=name)
        
        # Inputs: enc(rho_i) (T). Output: enc_state (Z). Hidden state: h_i: z = h_t (Z).
        self.encoder_gru = nn.GRU(self.num_program_tokens, self.hidden_size)
        init_gru(self.encoder_gru)
        
        # Inputs: enc(rho_i) (T), z (Z). Output: dec_state (Z). Hidden state: h_i: h_0 = z (Z).
        self.decoder_gru = nn.GRU(self.hidden_size + self.num_program_tokens, self.hidden_size)
        init_gru(self.decoder_gru)
        
        # Input: dec_state (Z), z (Z), enc(rho_i) (T). Output: prob(rho_hat) (T).
        self.decoder_mlp = nn.Sequential(
            self.init_(nn.Linear(2 * self.hidden_size + self.num_program_tokens, self.hidden_size)),
            nn.Tanh(), self.init_(nn.Linear(self.hidden_size, self.num_program_tokens))
        )
        
        # Input: enc(s_i) (Z), enc(a_i) (A), z (Z). Output: pol_out (Z).
        self.policy_gru = nn.GRU(2 * self.hidden_size + self.num_agent_actions, self.hidden_size)
        init_gru(self.policy_gru)
        
        # Inputs: pol_out (Z). Output: prob(a_hat) (A).
        self.policy_mlp = nn.Sequential(
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.num_agent_actions))
        )
        
        # Input: s_i (CxHxW). Output: enc(s_i) (Z).
        self.state_encoder = nn.Sequential(
            self.init_(nn.Conv2d(self.state_shape[0], 32, 3, stride=1)), nn.ReLU(),
            self.init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), nn.Flatten(),
            self.init_(nn.Linear(32 * 4 * 4, self.hidden_size)), nn.ReLU()
        )
        
        # Input: a_i (A). Output: enc(a_i) (A).
        self.action_encoder = nn.Embedding(self.num_agent_actions, self.num_agent_actions)
        
        # Input: rho_i (T). Output: enc(rho_i) (T).
        self.token_encoder = nn.Embedding(self.num_program_tokens, self.num_program_tokens)
        
        # Encoder VAE utils
        self.encoder_mu = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_log_sigma = torch.nn.Linear(self.hidden_size, self.hidden_size)
        
        self.to(self.device)

    def sample_latent_vector(self, enc_hidden_state: torch.Tensor) -> torch.Tensor:
        # Sampling z with reperameterization trick
        mu = self.encoder_mu(enc_hidden_state)
        log_sigma = self.encoder_log_sigma(enc_hidden_state)
        sigma = torch.exp(log_sigma)
        std_z = torch.randn(sigma.size(), device=self.device)
        
        z = mu + sigma * std_z
        
        self.z_mu = mu
        self.z_sigma = sigma
        
        return z
    
    def get_latent_loss(self):
        mean_sq = self.z_mu * self.z_mu
        stddev_sq = self.z_sigma * self.z_sigma
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
        
    def encode(self, progs: torch.Tensor, progs_mask: torch.Tensor):
        if len(progs.shape) == 3:
            batch_size, demos_per_program, _ = progs.shape
            progs = progs.view(batch_size * demos_per_program, -1)
            progs_mask = progs_mask.view(batch_size * demos_per_program, -1)
        
        progs_len = progs_mask.squeeze(-1).sum(dim=-1).cpu()
        
        enc_progs = self.token_encoder(progs)
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            enc_progs, progs_len, batch_first=True, enforce_sorted=False
        )
        
        _, enc_hidden_state = self.encoder_gru(packed_inputs)
        enc_hidden_state = enc_hidden_state.squeeze(0)
        
        z = self.sample_latent_vector(enc_hidden_state)
        
        return z
    
    def decode(self, z: torch.Tensor, progs: torch.Tensor, progs_mask: torch.Tensor,
               prog_teacher_enforcing = True):
        if progs is not None:
            if len(progs.shape) == 3:
                b, demos_per_program, _ = progs.shape
                progs = progs.view(b * demos_per_program, -1)
                progs_mask = progs_mask.view(b * demos_per_program, -1)
        
        batch_size, _ = z.shape
        
        gru_hidden_state = z.unsqueeze(0)
        
        # Initialize tokens as DEFs
        current_tokens = torch.zeros((batch_size), dtype=torch.long, device=self.device)
        
        grammar_state = [self.syntax_checker.get_initial_checker_state()
                         for _ in range(batch_size)]
        
        pred_progs = []
        pred_progs_logits = []
        
        for i in range(1, self.max_program_length):
            token_embedding = self.token_encoder(current_tokens)
            gru_inputs = torch.cat((token_embedding, z), dim=-1)
            gru_inputs = gru_inputs.unsqueeze(0)
            
            gru_output, gru_hidden_state = self.decoder_gru(gru_inputs, gru_hidden_state)
            
            mlp_input = torch.cat([gru_output.squeeze(0), token_embedding, z], dim=1)
            pred_token_logits = self.decoder_mlp(mlp_input)
            
            syntax_mask, grammar_state = self.get_syntax_mask(batch_size, current_tokens, grammar_state)
            
            pred_token_logits += syntax_mask
            
            pred_tokens = self.softmax(pred_token_logits).argmax(dim=-1)
            
            pred_progs.append(pred_tokens)
            pred_progs_logits.append(pred_token_logits)
            
            if prog_teacher_enforcing:
                # Enforce next token with ground truth
                current_tokens = progs[:, i].view(batch_size)
            else:
                # Pass current prediction to next iteration
                current_tokens = pred_tokens.view(batch_size)
        
        pred_progs = torch.stack(pred_progs, dim=1)
        pred_progs_logits = torch.stack(pred_progs_logits, dim=1)
        pred_progs_masks = (pred_progs != self.num_program_tokens - 1)
        
        return pred_progs, pred_progs_logits, pred_progs_masks
    
    def policy_executor(self, z: torch.Tensor, s_h: torch.Tensor, a_h: torch.Tensor,
                        a_h_mask: torch.Tensor, a_h_teacher_enforcing = True):
        batch_size, demos_per_program, _, c, h, w = s_h.shape
        
        # Taking only first state and squeezing over first 2 dimensions
        current_state = s_h[:, :, 0, :, :, :].view(batch_size*demos_per_program, c, h, w)
        
        ones = torch.ones((batch_size*demos_per_program, 1), dtype=torch.long, device=self.device)
        current_action = (self.num_agent_actions - 1) * ones
        
        z_repeated = z.unsqueeze(1).repeat(1, demos_per_program, 1)
        z_repeated = z_repeated.view(batch_size*demos_per_program, self.hidden_size)
        
        gru_hidden = z_repeated.unsqueeze(0)
        
        pred_a_h = []
        pred_a_h_logits = []
        
        if not a_h_teacher_enforcing:
            self.env_init(current_state)
        
        terminated_policy = torch.zeros_like(current_action, dtype=torch.bool, device=self.device)
        
        mask_valid_actions = torch.tensor((self.num_agent_actions - 1) * [-torch.finfo(torch.float32).max]
                                          + [0.], device=self.device)
        
        for i in range(1, self.max_demo_length):
            enc_state = self.state_encoder(current_state)
            
            enc_action = self.action_encoder(current_action.squeeze(-1))
            
            gru_inputs = torch.cat((z_repeated, enc_state, enc_action), dim=-1)
            gru_inputs = gru_inputs.unsqueeze(0)
            
            gru_out, gru_hidden = self.policy_gru(gru_inputs, gru_hidden)
            gru_out = gru_out.squeeze(0)
            
            pred_action_logits = self.policy_mlp(gru_out)
            
            masked_action_logits = pred_action_logits + terminated_policy * mask_valid_actions
            
            current_action = self.softmax(masked_action_logits).argmax(dim=-1).view(-1, 1)
            
            pred_a_h.append(current_action)
            pred_a_h_logits.append(pred_action_logits)
            
            # Apply teacher enforcing while training
            if a_h_teacher_enforcing:
                current_state = s_h[:, :, i, :, :, :].view(batch_size*demos_per_program, c, h, w)
                current_action = a_h[:, :, i].view(batch_size*demos_per_program, 1)
            # Otherwise, step in actual environment to get next state
            else:
                current_state = self.env_step(current_action)
                
            terminated_policy = torch.logical_or(current_action == self.num_agent_actions - 1,
                                                 terminated_policy)
    
        pred_a_h = torch.stack(pred_a_h, dim=1).squeeze(-1)
        pred_a_h_logits = torch.stack(pred_a_h_logits, dim=1)
        pred_a_h_masks = (pred_a_h != self.num_agent_actions - 1)
        
        return pred_a_h, pred_a_h_logits, pred_a_h_masks
    
    def forward(self, data_batch: tuple, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelReturn:
        s_h, a_h, a_h_masks, progs, progs_masks = data_batch
        
        z = self.encode(progs, progs_masks)
        
        pred_progs, pred_progs_logits, pred_progs_masks = self.decode(
            z, progs, progs_masks, prog_teacher_enforcing
        )
        pred_a_h, pred_a_h_logits, pred_a_h_masks = self.policy_executor(
            z, s_h, a_h, a_h_masks, a_h_teacher_enforcing
        )
        
        return ModelReturn(z, pred_progs, pred_progs_logits, pred_progs_masks,
                           pred_a_h, pred_a_h_logits, pred_a_h_masks)
        
    def encode_program(self, prog: torch.Tensor):
        if prog.dim() == 1:
            prog = prog.unsqueeze(0)
        
        prog_mask = (prog != self.num_program_tokens - 1)
        
        z = self.encode(prog, prog_mask)
        
        return z
    
    def decode_vector(self, z: torch.Tensor):
        pred_progs, _, pred_progs_masks = self.decode(z, None, None, False)
        
        pred_progs_tokens = []
        for prog, prog_mask in zip(pred_progs, pred_progs_masks):
            pred_progs_tokens.append([0] + prog[prog_mask].cpu().numpy().tolist())
        
        return pred_progs_tokens
    
    def get_losses_and_accs(self, data_batch: tuple, output: ModelReturn,
                                  loss_fn: nn.CrossEntropyLoss) -> tuple[Losses, Accuracies]:
        _, a_h, a_h_masks, progs, progs_masks = data_batch

        _, pred_progs, pred_progs_logits, pred_progs_masks,\
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
            # Combine first 2 dimensions of progs (batch_size and demos_per_program)
            progs = progs.view(-1, progs.shape[-1])
            progs_masks = progs_masks.view(-1, progs_masks.shape[-1])
            
            # Skip first token in ground truth sequences
            progs = progs[:, 1:].contiguous()
            progs_masks = progs_masks[:, 1:].contiguous()

            # Flatten everything for loss calculation
            progs_flat = progs.view(-1, 1)
            progs_masks_flat = progs_masks.view(-1, 1)
        
        if pred_a_h is not None:
            pred_a_h_logits = pred_a_h_logits.view(-1, pred_a_h_logits.shape[-1])
            pred_a_h_masks_flat = pred_a_h_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            a_h_masks_flat_combined = torch.max(a_h_masks_flat, pred_a_h_masks_flat).squeeze()
        
        if pred_progs is not None:
            pred_progs_logits = pred_progs_logits.view(-1, pred_progs_logits.shape[-1])
            pred_progs_masks_flat = pred_progs_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            progs_masks_flat_combined = torch.max(progs_masks_flat, pred_progs_masks_flat).squeeze()
        
        zero_tensor = torch.tensor([0.0], device=self.device, requires_grad=False)
        progs_loss, a_h_loss = zero_tensor, zero_tensor
        
        if pred_a_h is not None:
            a_h_loss = loss_fn(pred_a_h_logits[a_h_masks_flat_combined],
                               a_h_flat[a_h_masks_flat_combined].view(-1))
        
        if pred_progs is not None:
            progs_loss = loss_fn(pred_progs_logits[progs_masks_flat_combined],
                                 progs_flat[progs_masks_flat_combined].view(-1))

        latent_loss = self.get_latent_loss()
        
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
        
        return Losses(progs_loss, a_h_loss, latent_loss),\
            Accuracies(progs_t_accuracy, progs_s_accuracy, a_h_t_accuracy, a_h_s_accuracy)

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
            prog_loss_coef: float = 1.0, a_h_loss_coef: float = 1.0,
            latent_loss_coef: float = 0.1,
            disable_prog_teacher_enforcing: bool = False,
            disable_a_h_teacher_enforcing: bool = False,
            optim_lr: float = 5e-4, save_params_each_epoch: bool = False,
            num_epochs: int = 100, logger: Logger = None,
            base_output_folder: str = 'output'):
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=optim_lr
        )
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        output_folder = os.path.join(base_output_folder, 'trainer', self.name)
        model_folder = os.path.join(output_folder, 'model')
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(model_folder, exist_ok=True)
        
        if val_dataloader is not None:
            best_val_return = float('inf')
        
        training_info_csv = os.path.join(output_folder, 'training_info.csv')
        validation_info_csv = os.path.join(output_folder, 'validation_info.csv')
        
        fields = ['total_loss'] + [loss + '_loss' for loss in Losses._fields] +\
            [acc + '_accuracy' for acc in Accuracies._fields]
        
        with open(training_info_csv, 'w') as f:
            f.write('epoch,')
            f.write(','.join(fields))
            f.write('\n')
            
        if val_dataloader is not None:
            with open(validation_info_csv, 'w') as f:
                f.write('epoch,')
                f.write(','.join(fields))
                f.write('\n')
        
        for epoch in range(1, num_epochs + 1):
            self.log(f'Training epoch {epoch}/{num_epochs}')
            
            self.train()
            torch.set_grad_enabled(True)
            
            train_batches_total_losses = np.zeros((len(train_dataloader), 1))
            train_batches_losses = np.zeros((len(train_dataloader), len(Losses._fields)))
            train_batches_accs = np.zeros((len(train_dataloader), len(Accuracies._fields)))
            
            # Training step
            for i, train_batch in enumerate(train_dataloader):
                
                optimizer.zero_grad()

                output = self(train_batch,
                              not disable_prog_teacher_enforcing,
                              not disable_a_h_teacher_enforcing)
                losses, accs = self.get_losses_and_accs(train_batch, output, loss_fn)
                
                total_loss = prog_loss_coef * losses.progs_rec +\
                    a_h_loss_coef * losses.a_h_rec +\
                    latent_loss_coef * losses.latent
                
                total_loss.backward()
                optimizer.step()
                
                train_batches_total_losses[i] = total_loss.item()
                train_batches_losses[i] = np.array([loss.item() for loss in losses])
                train_batches_accs[i] = np.array([acc.item() for acc in accs])
                
            train_mean_total_loss = train_batches_total_losses.mean()
            train_mean_losses = train_batches_losses.mean(axis=0)
            train_mean_accs = train_batches_accs.mean(axis=0)
            
            self.log('Total loss: ' + str(train_mean_total_loss))
            self.log(str(Losses(*train_mean_losses.tolist())))
            self.log(str(Accuracies(*train_mean_accs.tolist())))
            
            with open(training_info_csv, 'a') as f:
                f.write(f'{epoch},{train_mean_total_loss},')
                f.write(','.join([str(loss) for loss in train_mean_losses]))
                f.write(',')
                f.write(','.join([str(acc) for acc in train_mean_accs]))
                f.write('\n')
                
            if val_dataloader is not None:
                self.log(f'Validation epoch {epoch}/{num_epochs}')
                
                self.eval()
                torch.set_grad_enabled(False)
                
                val_batches_total_losses = np.zeros((len(val_dataloader), 1))
                val_batches_losses = np.zeros((len(val_dataloader), len(Losses._fields)))
                val_batches_accs = np.zeros((len(val_dataloader), len(Accuracies._fields)))
                
                # Validation step
                for i, val_batch in enumerate(val_dataloader):
                    
                    output = self(val_batch,
                                  not disable_prog_teacher_enforcing,
                                  not disable_a_h_teacher_enforcing)
                    losses, accs = self.get_losses_and_accs(val_batch, output, loss_fn)
                    
                    total_loss = prog_loss_coef * losses.progs_rec +\
                        a_h_loss_coef * losses.a_h_rec +\
                        latent_loss_coef * losses.latent
                    
                    val_batches_total_losses[i] = total_loss.item()
                    val_batches_losses[i] = np.array([loss.item() for loss in losses])
                    val_batches_accs[i] = np.array([acc.item() for acc in accs])

                val_mean_total_loss = val_batches_total_losses.mean()
                val_mean_losses = val_batches_losses.mean(axis=0)
                val_mean_accs = val_batches_accs.mean(axis=0)

                self.log('Total loss: ' + str(val_mean_total_loss))
                self.log(str(Losses(*val_mean_losses.tolist())))
                self.log(str(Accuracies(*val_mean_accs.tolist())))
                    
                with open(validation_info_csv, 'a') as f:
                    f.write(f'{epoch},{val_mean_total_loss},')
                    f.write(','.join([str(loss) for loss in val_mean_losses]))
                    f.write(',')
                    f.write(','.join([str(acc) for acc in val_mean_accs]))
                    f.write('\n')
                    
                if val_mean_total_loss < best_val_return:
                    self.log(f'New best validation loss: {val_mean_total_loss}')
                    best_val_return = val_mean_total_loss
                    params_path = os.path.join(model_folder, f'best_val.ptp')
                    torch.save(self.state_dict(), params_path)
                    self.log(f'Parameters saved to {params_path}')
