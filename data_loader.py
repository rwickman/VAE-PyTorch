import torch
from torch.utils.data import Dataset

def test_dataset(mean = 5, std = 2, dataset_size=512, input_dim=10):
    return std * torch.randn(size=[input_dim]) + torch.tensor(mean).repeat(input_dim)


class TestDataset(Dataset):
    def __init__(self):
        std = 2
        mean = 5
        input_dim = 10
        self.ds = std * torch.randn(size=[1024, input_dim]) + torch.tensor(mean).repeat(input_dim)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
    
        return self.ds[idx]