from torch.utils.data import Dataset
import torch

class BatchDataSet(Dataset):
    def __init__(self, batch, device):
        self.dataset = batch
        self.device = device
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self.obs = torch.tensor(self.dataset[idx].obs, device=self.device, dtype=torch.float32)
        self.obs_next = torch.tensor(self.dataset[idx].obs_next, device=self.device, dtype=torch.float32)
        self.act = torch.tensor(self.dataset[idx].act, device=self.device, dtype=torch.int64)
        return {'obs':self.obs,'obs_next':self.obs_next,'act':self.act}