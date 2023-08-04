import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from prog_policies.base import BaseDSL

class ProgramDataset(Dataset):

    def __init__(self, program_list: list, dsl: BaseDSL, device: torch.device,
                 max_program_length = 45, max_demo_length = 100):
        self.device = device
        self.programs = program_list
        # need this +1 as DEF token is input to decoder, loss will be calculated only from run token
        self.max_program_len = max_program_length + 1
        self.max_demo_length = max_demo_length
        self.parse_str_to_int = dsl.parse_str_to_int
        self.pad_token = dsl.t2i['<pad>']
        # NOP action is always one more the last in the list of actions
        self.action_nop = len(dsl.get_actions())

    def __len__(self):
        return len(self.programs)

    def __getitem__(self, idx):
        prog_str, (s_h, a_h, a_h_len) = self.programs[idx]
        
        prog_int = self.parse_str_to_int(prog_str)
        prog = np.array(prog_int)

        prog = torch.from_numpy(prog).to(self.device).to(torch.long)
        program_len = prog.shape[0]
        prog_sufix = torch.tensor((self.max_program_len - program_len - 1) * [self.pad_token],
                                  device=self.device, dtype=torch.long)
        prog = torch.cat((prog, prog_sufix))
        
        a_h_expanded = np.ones((a_h.shape[0], self.max_demo_length), dtype=int) * (self.action_nop)
        s_h_expanded = np.zeros((s_h.shape[0], self.max_demo_length, *s_h.shape[2:]), dtype=bool)

        # Add no-op actions for empty demonstrations
        for i in range(a_h_len.shape[0]):
            a_h_expanded[i, 1:a_h_len[i]+1] = a_h[i, :a_h_len[i]]
            s_h_expanded[i, :a_h_len[i]+1] = s_h[i, :a_h_len[i]+1]
            s_h_expanded[i, a_h_len[i]+1:] = s_h[i, a_h_len[i]] * (self.max_demo_length - a_h_len[i] + 1)
        
        s_h = torch.tensor(s_h_expanded, device=self.device, dtype=torch.float32)
        a_h = torch.tensor(a_h_expanded, device=self.device, dtype=torch.long)

        prog_mask = (prog != self.pad_token)
        a_h_mask = (a_h != self.action_nop)

        return s_h, a_h, a_h_mask, prog, prog_mask


class ProgramPerceptionsDataset(ProgramDataset):
    
    def __getitem__(self, idx):
        prog_str, (_, perc_h, a_h, a_h_len) = self.programs[idx]
        
        prog_int = self.parse_str_to_int(prog_str)
        prog = np.array(prog_int)

        prog = torch.from_numpy(prog).to(self.device).to(torch.long)
        program_len = prog.shape[0]
        prog_sufix = torch.tensor((self.max_program_len - program_len - 1) * [self.pad_token],
                                  device=self.device, dtype=torch.long)
        prog = torch.cat((prog, prog_sufix))
        
        a_h_expanded = np.ones((a_h.shape[0], self.max_demo_length), dtype=int) * (self.action_nop)
        perc_h_expanded = np.zeros((perc_h.shape[0], self.max_demo_length, *perc_h.shape[2:]), dtype=bool)

        # Add no-op actions for empty demonstrations
        for i in range(a_h_len.shape[0]):
            a_h_expanded[i, 1:a_h_len[i]+1] = a_h[i, :a_h_len[i]]
            perc_h_expanded[i, :a_h_len[i]+1] = perc_h[i, :a_h_len[i]+1]
            perc_h_expanded[i, a_h_len[i]+1:] = perc_h[i, a_h_len[i]] * (self.max_demo_length - a_h_len[i] + 1)
        
        perc_h = torch.tensor(perc_h_expanded, device=self.device, dtype=torch.float32)
        a_h = torch.tensor(a_h_expanded, device=self.device, dtype=torch.long)

        prog_mask = (prog != self.pad_token)
        a_h_mask = (a_h != self.action_nop)

        return perc_h, a_h, a_h_mask, prog, prog_mask


class ProgramOnlyDataset(ProgramDataset):
    
    def __getitem__(self, idx):
        prog_str, _ = self.programs[idx]

        prog_int = self.parse_str_to_int(prog_str)
        prog = torch.from_numpy(np.array(prog_int)).to(self.device).to(torch.long)
        program_len = prog.shape[0]
        prog_sufix = torch.tensor((self.max_program_len - program_len - 1) * [self.pad_token],
                                  device=self.device, dtype=torch.long)
        prog = torch.cat((prog, prog_sufix))

        prog_mask = (prog != self.pad_token)

        return [], [], [], prog, prog_mask


def make_dataloaders(dsl: BaseDSL, device: torch.device, dataset_path: str, data_class_name = 'ProgramDataset',
                     max_program_length = 45, max_demo_length = 100, batch_size = 32,
                     train_ratio = 0.8, val_ratio = 0.1, data_seed = 1):
    
    with open(dataset_path, 'rb') as f:
        program_list = pickle.load(f)
    
    rng = np.random.RandomState(data_seed)
    rng.shuffle(program_list)

    data_cls = globals()[data_class_name]
    assert issubclass(data_cls, Dataset)
    
    split_idx1 = int(train_ratio * len(program_list))
    split_idx2 = int((train_ratio + val_ratio)*len(program_list))
    train_program_list = program_list[:split_idx1]
    valid_program_list = program_list[split_idx1:split_idx2]
    test_program_list = program_list[split_idx2:]

    train_dataset = data_cls(train_program_list, dsl, device, max_program_length, max_demo_length)
    val_dataset = data_cls(valid_program_list, dsl, device, max_program_length, max_demo_length)
    test_dataset = data_cls(test_program_list, dsl, device, max_program_length, max_demo_length)
    
    torch_rng = torch.Generator().manual_seed(data_seed)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch_rng)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch_rng)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch_rng)
    
    return train_dataloader, val_dataloader, test_dataloader
