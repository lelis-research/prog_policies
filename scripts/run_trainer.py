import torch
import sys

sys.path.append('.')

from prog_policies.karel import KarelDSL, KarelEnvironment
from prog_policies.latent_space.models import LeapsVAE
from prog_policies.latent_space.program_dataset import make_dataloaders
from prog_policies.latent_space.trainer import Trainer
from prog_policies.output_handler import OutputHandler
from prog_policies.args_handler import parse_args

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env_args = {
        'crashable': False,
        'leaps_behaviour': True
    }
    model_args = {
        'hidden_size': args.model_hidden_size,
        'max_demo_length': args.data_max_demo_length,
        'max_program_length': args.data_max_program_length,
    }
    
    dsl = KarelDSL()
    env = KarelEnvironment(**env_args)
    output = OutputHandler(
        experiment_name=args.experiment_name
    )
    model = LeapsVAE(dsl, env, device, **model_args)
    dataloader_params = {
        'data_class_name': args.data_class_name,
        'dataset_path': args.data_program_dataset_path,
        'max_demo_length': args.data_max_demo_length,
        'max_program_length': args.data_max_program_length,
        'batch_size': args.data_batch_size,
        'train_ratio': args.data_ratio_train,
        'val_ratio': args.data_ratio_val,
        'random_seed': args.env_seed
    }
    train_data, val_data, _ = make_dataloaders(dsl, device, **dataloader_params)
    trainer_params = {
        'num_epochs': args.trainer_num_epochs,
        'prog_loss_coef': args.trainer_prog_loss_coef,
        'a_h_loss_coef': args.trainer_a_h_loss_coef,
        'latent_loss_coef': args.trainer_latent_loss_coef,
        'disable_prog_teacher_enforcing': args.trainer_disable_prog_teacher_enforcing,
        'disable_a_h_teacher_enforcing': args.trainer_disable_a_h_teacher_enforcing,
        'optim_lr': args.trainer_optim_lr,
        'save_params_each_epoch': args.trainer_save_params_each_epoch,
        'output_handler': output,
    }
    trainer = Trainer(model, **trainer_params)
    trainer.train(train_data, val_data)
