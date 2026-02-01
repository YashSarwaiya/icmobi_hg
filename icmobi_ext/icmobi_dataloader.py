import torch
from torch.utils.data import Dataset
import pandas as pd
import torch
import random

#%% TOPOGRAPHY TRANSFORMS
class RandomHorizontalFlipTopography:
    """Flip topography along y-axis with 50% chance."""
    def __call__(self, sample):
        if random.random() > 0.5:
            sample['topo'] = torch.flip(sample['topo'], dims=[-1])
        return sample

class RandomGaussianNoiseTopography:
    """Add random Gaussian noise to topography."""
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        noise = torch.randn_like(sample['topo']) * self.std + self.mean
        sample['topo'] = sample['topo'] + noise
        return sample

class NormalizeTopography:
    """Z-score normalize topography map."""
    def __call__(self, sample):
        topo = sample['topo']
        mean = topo.mean()
        std = topo.std()
        sample['topo'] = (topo - mean) / (std + 1e-8)  # Prevent divide by zero
        return sample

class RandomTopographyDropout:
    """Randomly mask a rectangular patch (cutout-style)."""
    def __init__(self, max_size=8):
        self.max_size = max_size

    def __call__(self, sample):
        topo = sample['topo']
        _, h, w = topo.shape

        size = random.randint(4, self.max_size)
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        topo[:, y:y+size, x:x+size] = 0.0
        sample['topo'] = topo
        return sample
    
#%% PSD TRANSFORMS
class NormalizePSD:
    """Z-score normalize PSD feature vector."""
    def __call__(self, sample):
        psd = sample['psd']
        mean = psd.mean()
        std = psd.std()
        sample['psd'] = (psd - mean) / (std + 1e-8)
        return sample
    
class AddNoisePSD:
    """Add Gaussian noise to PSD vector."""
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, sample):
        sample['psd'] = sample['psd'] + torch.randn_like(sample['psd']) * self.std
        return sample

#%% AC TRANSFORMS
class NormalizeAC:
    """Z-score normalize autocorrelation feature vector."""
    def __call__(self, sample):
        ac = sample['ac']
        mean = ac.mean()
        std = ac.std()
        sample['ac'] = (ac - mean) / (std + 1e-8)
        return sample

class AddNoiseAC:
    """Add Gaussian noise to AC vector."""
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, sample):
        sample['ac'] = sample['ac'] + torch.randn_like(sample['ac']) * self.std
        return sample


#%% DATASET
class ICMOBIDatasetFormatter(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None,
                 psd_regexp="PSD",ac_regexp="AC",topo_regexp="TOPO",label_regexp="LABEL"):
        self.transform  = transform
        self.df_len = len(df)
        
        # self.psd_cols = [f"{psd_regexp}_{i}" for i in range(1, 101)]
        # self.ac_cols = [f"{ac_regexp}_{i}" for i in range(1, 101)]
        # self.topo_cols = [f"{topo_regexp}_{i}" for i in range(1, 1025)]
        # self.label_cols = [f"{label_regexp}_{i}" for i in range(1, 8)]
        
        self.psd_cols = [col for col in df.columns if col.startswith(f"{psd_regexp}_")]
        self.ac_cols = [col for col in df.columns if col.startswith(f"{ac_regexp}_")]
        self.topo_cols = [col for col in df.columns if col.startswith(f"{topo_regexp}_")]
        self.label_cols = [col for col in df.columns if col.startswith(f"{label_regexp}_")]
        
        self.psd = torch.tensor(df[self.psd_cols].values, dtype=torch.float32).unsqueeze(1).unsqueeze(2)  # (N, 1, 100)
        self.ac = torch.tensor(df[self.ac_cols].values, dtype=torch.float32).unsqueeze(1).unsqueeze(2)    # (N, 1, 100)
        self.topo = torch.tensor(df[self.topo_cols].values, dtype=torch.float32).view(-1, 1, 32, 32)  # (N, 1, 32, 32)
        # self.topo = torch.tensor(df[self.topo_cols].values, dtype=torch.float32).unsqueeze(1).unsqueeze(2)  # (N, 1, 32, 32)
        self.labels = torch.tensor(df[self.label_cols].values, dtype=torch.float32)

    def __len__(self):
        return self.df_len
    "end"    

    def __getitem__(self, idx):        
        sample = {
            'psd': self.psd[idx].clone(),
            'ac': self.ac[idx].clone(),
            'topo': self.topo[idx].clone(),
            'label': self.labels[idx].clone()
        }
        
        # sample = (torch.tensor(self.topo[idx], dtype=torch.float32),                                      
        #         torch.tensor(self.psd[idx], dtype=torch.float32),
        #         torch.tensor(self.ac[idx], dtype=torch.float32),
        #         torch.tensor(self.labels[idx], dtype=torch.float32))
        
        #-- Apply horizontal flip to topography with 50% probability
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    "end"
"end"

    