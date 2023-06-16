import torch
import sys

sys.path.append('.')

from prog_policies.karel import KarelDSL, KarelEnvironment
from prog_policies.latent_space.models import LeapsVAE
from prog_policies.latent_space.program_dataset import make_dataloaders
from prog_policies.latent_space.trainer import Trainer
from prog_policies.output_handler import OutputHandler
from prog_policies.args_handler import parse_args
from prog_policies.config import Config

if __name__ == '__main__':
    parse_args()

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env_args = {
        'crashable': False,
        'leaps_behaviour': True
    }
    model_args = {
        'hidden_size': Config.model_hidden_size,
        'max_demo_length': Config.data_max_demo_length,
        'max_program_length': Config.data_max_program_length,
    }
    
    dsl = KarelDSL()
    env = KarelEnvironment(**env_args)
    output = OutputHandler(
        experiment_name=Config.experiment_name
    )
    model = LeapsVAE(dsl, env, device, **model_args)
    dataloader_params = {
        'data_class_name': Config.data_class_name,
        'dataset_path': Config.data_program_dataset_path,
        'max_demo_length': Config.data_max_demo_length,
        'max_program_length': Config.data_max_program_length,
        'batch_size': Config.data_batch_size,
        'train_ratio': Config.data_ratio_train,
        'val_ratio': Config.data_ratio_val,
        'random_seed': Config.env_seed
    }
    train_data, val_data, _ = make_dataloaders(dsl, device, **dataloader_params)
    trainer_params = {
        'num_epochs': Config.trainer_num_epochs,
        'prog_loss_coef': Config.trainer_prog_loss_coef,
        'a_h_loss_coef': Config.trainer_a_h_loss_coef,
        'latent_loss_coef': Config.trainer_latent_loss_coef,
        'disable_prog_teacher_enforcing': Config.trainer_disable_prog_teacher_enforcing,
        'disable_a_h_teacher_enforcing': Config.trainer_disable_a_h_teacher_enforcing,
        'optim_lr': Config.trainer_optim_lr,
        'save_params_each_epoch': Config.trainer_save_params_each_epoch,
        'output_handler': output,
    }
    trainer = Trainer(model, **trainer_params)
    trainer.train(train_data, val_data)
