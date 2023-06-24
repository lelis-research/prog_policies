from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--disable_gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--experiment_name', default='leaps_vae_test', help='Name of the experiment')
    parser.add_argument('--log_filename', default='leaps_vae_debugging', help='Name of the log file')
    parser.add_argument('--multiprocessing_active', action='store_true', help='If set, search functions will use multiprocessing to evaluate programs')

    parser.add_argument('--model_name', default='LeapsVAE', help='Class name of the VAE model')
    parser.add_argument('--model_seed', default=1, type=int, help='Seed for model initialization')
    parser.add_argument('--model_hidden_size', default=256, type=int, help='Number of dimensions in VAE hidden unit')
    parser.add_argument('--model_params_path', default='params/leaps_vae_256.ptp', help='Path to model parameters')
    
    parser.add_argument('--datagen_num_programs', default=50000, type=int, help='Number of programs in dataset, used for data generation and loading')
    parser.add_argument('--datagen_sketch_iterations', default=3, type=int, help='Number of needed Top-Down iterations to reconstruct a program from its sketch')
    parser.add_argument('--datagen_generate_demos', action='store_true', help='If set, generates demonstrations for each program')
    parser.add_argument('--datagen_generate_sketches', action='store_true', help='If set, generates sketches for each program')
    
    parser.add_argument('--data_class_name', default='ProgramDataset', help='Name of program dataset class')
    parser.add_argument('--data_program_dataset_path', default='data/leaps_dataset.pkl', help='Path to program dataset')
    parser.add_argument('--data_batch_size', default=256, type=int, help='Batch size used in VAE training')
    parser.add_argument('--data_max_program_length', default=45, type=int, help='Maximum program length in number of tokens')
    parser.add_argument('--data_max_program_size', default=20, type=int, help='Maximum program size in number of nodes')
    parser.add_argument('--data_max_program_depth', default=4, type=int, help='Max allowed depth during program generation')
    parser.add_argument('--data_max_program_sequence', default=6, type=int, help='Max allowed number of sequential nodes aggregated by Concatenate')
    parser.add_argument('--data_max_demo_length', default=20, type=int, help='Maximum action history length in number of actions')
    parser.add_argument('--data_num_demo_per_program', default=10, type=int, help='Number of demonstrations per program in dataset')
    parser.add_argument('--data_ratio_train', default=0.7, type=float, help='Ratio of training data')
    parser.add_argument('--data_ratio_val', default=0.15, type=float, help='Ratio of validation data')
    parser.add_argument('--data_ratio_test', default=0.15, type=float, help='Ratio of test data')
    
    parser.add_argument('--env_task', default='StairClimber', help='Name of Karel task to solve')
    parser.add_argument('--env_seed', default=1, type=int, help='Seed for random environment generation')
    parser.add_argument('--env_height', default=8, type=int, help='Height of Karel environment')
    parser.add_argument('--env_width', default=8, type=int, help='Width of Karel environment')
    parser.add_argument('--env_enable_leaps_behaviour', action='store_true', help='If set, uses LEAPS version of Karel rules')
    parser.add_argument('--env_is_crashable', action='store_true', help='If set, program stops when Karel crashes')
    
    parser.add_argument('--search_method', default='LatentCEM', help='Name of search method class')
    parser.add_argument('--search_args_path', default='sample_args/search/latent_cem.json', help='Arguments path for search method')

    parser.add_argument('--trainer_num_epochs', default=150, type=int, help='Number of training epochs')
    parser.add_argument('--trainer_disable_prog_teacher_enforcing', action='store_true', help='If set, program sequence classification will not use teacher enforcing')
    parser.add_argument('--trainer_disable_a_h_teacher_enforcing', action='store_true', help='If set, actions sequence classification will not use teacher enforcing')
    parser.add_argument('--trainer_prog_loss_coef', default=1.0, type=float, help='Weight of program classification loss')
    parser.add_argument('--trainer_a_h_loss_coef', default=1.0, type=float, help='Weight of actions classification loss')
    parser.add_argument('--trainer_latent_loss_coef', default=0.1, type=float, help='Weight of VAE KL Divergence Loss')
    parser.add_argument('--trainer_optim_lr', default=5e-4, type=float, help='Adam optimizer learning rate')
    parser.add_argument('--trainer_save_params_each_epoch', action='store_true', help='If set, trainer saves model params after each epoch')
    
    return parser.parse_args()
