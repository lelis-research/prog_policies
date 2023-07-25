import logging
import torch
import unittest, sys

sys.path.append('.')

from prog_policies.karel import KarelDSL, KarelEnvironment
from prog_policies.latent_space.models import LeapsVAE
from prog_policies.latent_space.syntax_checker import SyntaxChecker
from prog_policies.latent_space.program_dataset import make_dataloaders

class TestLatentSpace(unittest.TestCase):
    
    def testSyntaxChecker(self):
        dsl = KarelDSL()
        sample_program = [
            dsl.t2i['DEF'],
            dsl.t2i['run'],
            dsl.t2i['m('],
            
            dsl.t2i['IF'],
            dsl.t2i['c('],
            dsl.t2i['frontIsClear'],
            dsl.t2i['c)'],
            
            dsl.t2i['i('],
            dsl.t2i['move'],
            dsl.t2i['turnLeft'],
            dsl.t2i['i)'],
            
            dsl.t2i['turnLeft'],
            
            dsl.t2i['m)']
        ]

        syntax_checker = SyntaxChecker(dsl, torch.device('cpu'), only_structure=False)
        initial_state = syntax_checker.get_initial_checker_state()
        sequence_mask = syntax_checker.get_sequence_mask(initial_state, sample_program).squeeze()
        valid_tokens_list = []
        for idx, _ in enumerate(sample_program):
            valid_tokens = torch.where(sequence_mask[idx] == 0)[0].cpu().numpy().tolist()
            valid_tokens_list.append(valid_tokens)
        
        # Mandatory token to start program
        self.assertSetEqual(set(valid_tokens_list[1]), set([dsl.t2i['m(']]))
        # Check if checker does not allow empty program
        self.assertSetEqual(set(valid_tokens_list[2]), set([dsl.t2i['IF'],
                                                            dsl.t2i['WHILE'],
                                                            dsl.t2i['IFELSE'],
                                                            dsl.t2i['REPEAT'],
                                                            dsl.t2i['move'],
                                                            dsl.t2i['turnLeft'],
                                                            dsl.t2i['turnRight'],
                                                            dsl.t2i['pickMarker'],
                                                            dsl.t2i['putMarker']]))
        # Check if there are only boolean tokens in the condition
        self.assertSetEqual(set(valid_tokens_list[4]), set([dsl.t2i['not'],
                                                            dsl.t2i['frontIsClear'],
                                                            dsl.t2i['rightIsClear'],
                                                            dsl.t2i['leftIsClear'],
                                                            dsl.t2i['markersPresent'],
                                                            dsl.t2i['noMarkersPresent']]))
        # Check if checker does not allow empty statement
        self.assertSetEqual(set(valid_tokens_list[7]), set([dsl.t2i['IF'],
                                                            dsl.t2i['WHILE'],
                                                            dsl.t2i['IFELSE'],
                                                            dsl.t2i['REPEAT'],
                                                            dsl.t2i['move'],
                                                            dsl.t2i['turnLeft'],
                                                            dsl.t2i['turnRight'],
                                                            dsl.t2i['pickMarker'],
                                                            dsl.t2i['putMarker']]))
        # Checker should now allow the end of the if statement
        self.assertSetEqual(set(valid_tokens_list[8]), set([dsl.t2i['IF'],
                                                            dsl.t2i['i)'],
                                                            dsl.t2i['WHILE'],
                                                            dsl.t2i['IFELSE'],
                                                            dsl.t2i['REPEAT'],
                                                            dsl.t2i['move'],
                                                            dsl.t2i['turnLeft'],
                                                            dsl.t2i['turnRight'],
                                                            dsl.t2i['pickMarker'],
                                                            dsl.t2i['putMarker']]))
        # Checker should now allow the end of the program
        self.assertSetEqual(set(valid_tokens_list[10]), set([dsl.t2i['m)'],
                                                            dsl.t2i['IF'],
                                                            dsl.t2i['WHILE'],
                                                            dsl.t2i['IFELSE'],
                                                            dsl.t2i['REPEAT'],
                                                            dsl.t2i['move'],
                                                            dsl.t2i['turnLeft'],
                                                            dsl.t2i['turnRight'],
                                                            dsl.t2i['pickMarker'],
                                                            dsl.t2i['putMarker']]))
    
    def testTrainer(self):
        logger = logging.getLogger()
        logger.disabled = True
        device = torch.device('cpu')
        dsl = KarelDSL()
        model_params = {
            'name': 'LeapsDebug',
            'env_args': {
                'env_width': 8,
                'env_height': 8,
                'crashable': False,
                'leaps_behaviour': True
            },
            'hidden_size': 16,
            'max_program_length': 45,
            'max_demo_length': 20,
            'model_seed': 1
        }
        model = LeapsVAE(dsl, device, KarelEnvironment, logger=logger, **model_params)
        dataloader_params = {
            'dataset_path': 'data/leaps_dataset_reduced.pkl',
            'batch_size': 16,
            'max_demo_length': 20,
        }
        train_data, val_data, _ = make_dataloaders(dsl, device, **dataloader_params)
        trainer_params = {
            'num_epochs': 1
        }
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        first_batch = next(iter(train_data))
        initial_output = model(first_batch)
        _, initial_accs = model.get_losses_and_accs(first_batch, initial_output, loss_fn)
        # Checking if our trainer can run a single epoch without errors
        model.fit(train_data, val_data, **trainer_params)        
        final_output = model(first_batch)
        _, final_accs = model.get_losses_and_accs(first_batch, final_output, loss_fn)
        # Checking if our accuracy improved at least on training data
        self.assertGreater(final_accs.progs_t, initial_accs.progs_t)
        self.assertGreater(final_accs.a_h_t, initial_accs.a_h_t)

if __name__ == '__main__':
    unittest.main()
