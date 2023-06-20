from datetime import datetime
import logging
import os
import sys
import torch

search_default_fields = ['time', 'num_evaluations', 'best_reward', 'best_program']

class OutputHandler:
    
    def __init__(self, experiment_name: str, log_filename: str = None):
        if log_filename is None:
            log_filename = experiment_name
        self.output_folder = os.path.join('output', experiment_name)
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.trainer_folder = os.path.join(self.output_folder, 'trainer')
        self.model_folder = os.path.join(self.output_folder, 'model')
        self.search_folder = os.path.join(self.output_folder, 'search')
        log_folder = os.path.join(self.output_folder, 'log')
        os.makedirs(log_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_folder, f'{log_filename}_{timestamp}.txt'), mode='w')
        ]
        
        logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.INFO)
        self.logger = logging.getLogger()
    
    def setup_trainer(self):
        os.makedirs(self.trainer_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        
    def setup_search(self):
        os.makedirs(self.search_folder, exist_ok=True)
        
    def setup_training_info(self, fields: list):
        with open(os.path.join(self.trainer_folder, 'training_info.csv'), mode='w') as f:
            f.write("epoch,")
            f.write(",".join(fields))
            f.write("\n")
    
    def setup_validation_info(self, fields: list):
        with open(os.path.join(self.trainer_folder, 'validation_info.csv'), mode='w') as f:
            f.write("epoch,")
            f.write(",".join(fields))
            f.write("\n")
    
    def save_training_info(self, epoch: int, info: list):
        with open(os.path.join(self.trainer_folder, 'training_info.csv'), mode='a') as f:
            f.write(f"{epoch},")
            f.write(",".join([str(i) for i in info]))
            f.write("\n")
    
    def save_validation_info(self, epoch: int, info: list):
        with open(os.path.join(self.trainer_folder, 'validation_info.csv'), mode='a') as f:
            f.write(f"{epoch},")
            f.write(",".join([str(i) for i in info]))
            f.write("\n")
    
    def save_model_parameters(self, model_name: str, model):
        params_path = os.path.join(self.model_folder, f'{model_name}.ptp')
        torch.save(model.state_dict(), params_path)
    
    def setup_search_info(self, search_seed: int, task_name: str, extra_fields: list = []):
        folder = os.path.join(self.search_folder, task_name)
        os.makedirs(folder, exist_ok=True)
        self.search_file = os.path.join(folder, f'seed_{search_seed}.csv')
        with open(self.search_file, mode='w') as f:
            f.write(",".join(search_default_fields + extra_fields))
            f.write("\n")
    
    def save_search_info(self, time, num_evaluations, best_reward, best_program,
                         extra_fields: list = []):
        with open(self.search_file, mode='a') as f:
            f.write(f"{time},{num_evaluations},{best_reward},{best_program}")
            f.write(",".join([str(i) for i in extra_fields]))
            f.write("\n")
    
    def log(self, sender: str, message: str):
        self.logger.info(f'[{sender}] {message}')
