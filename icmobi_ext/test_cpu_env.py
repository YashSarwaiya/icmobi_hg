#%% IMPORTS
import torch
import torch.distributed as dist
import torch.nn as nn
# import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
# from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
# from torch.multiprocessing import Pool, set_start_method
# import torch.multiprocessing as mp
from tqdm import tqdm
#--
from icmobi_ext import icmobi_model
from icmobi_ext import icmobi_dataloader
import pandas as pd
import os
# from icmobi_model import icldata, icmobi_model, plot_functions  # code adapted from https://github.com/lucapton/ICLabel-Dataset

#%% CPU INITIALIZATION
def setup_distributed(backend="gloo"):
    """Initialize distributed environment under SLURM + OpenMPI."""
    if "SLURM_PROCID" not in os.environ:
        raise EnvironmentError("This script must be run inside a SLURM job.")

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    print(f"[Rank {rank}] Initialized process group (world size={world_size})")

    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()
#enddef

#-- setup distributed computing for openmpi
rank, world_size, local_rank = setup_distributed("mpi")
device = torch.device("cpu")
#%% MULTIPROCESSING/THREADING
# Set the number of threads for various libraries
# os.environ["OMP_NUM_THREADS"] = "6"  # OpenMP (used by many scientific libraries)
# os.environ["OPENBLAS_NUM_THREADS"] = "6"
# os.environ["MKL_NUM_THREADS"] = "6"  # Intel MKL (used by NumPy/SciPy if linked)
# os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
# os.environ["NUMEXPR_NUM_THREADS"] = "6"

print(torch.__config__.parallel_info())
# import timeit
# runtimes = []
# threads = [1] + [t for t in range(2, 49, 2)]
# for t in threads:
#     torch.set_num_threads(t)
#     r = timeit.timeit(setup = "import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)", stmt="torch.mm(x, y)", number=100)
#     runtimes.append(r)
# # ... plotting (threads, runtimes) ...

# # Set up the distributed environment
# if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
#     rank = int(os.environ["SLURM_PROCID"])
#     world_size = int(os.environ["SLURM_NTASKS"])
# else:
#     raise RuntimeError("SLURM environment variables not found.")
# dist.init_process_group(backend="mpi", rank=rank, world_size=world_size)
# print(f"Hello from rank {rank} of world size {world_size}")

#%% DATASET TRANSFORMS
# transform_pipeline = transforms.Compose([
#     icmobi_dataloader.RandomHorizontalFlipTopography(),
#     icmobi_dataloader.RandomGaussianNoiseTopography(std=0.02),
#     icmobi_dataloader.RandomTopographyDropout(max_size=6),
#     icmobi_dataloader.NormalizeTopography(),
    
#     icmobi_dataloader.AddNoisePSD(std=0.01),
#     icmobi_dataloader.NormalizePSD(),

#     icmobi_dataloader.AddNoiseAC(std=0.01),
#     icmobi_dataloader.NormalizeAC()
# ])
transform_pipeline = None
#(07/14/2025) JS, torchvision doesn't work because of version 
# mismatch with PyTorch. I think its mainly because of the 
# Windows, but should to use version pairs: python setup.py install  # inside torchvision source dir
# or pip install torch==2.2.0 torchvision==0.17.0

#%% TEST DATALOADER
# #-- Load your pre-processed DataFrame
# rel_path = os.path.abspath("..\\data\\")
# df = pd.read_csv(os.path.join(rel_path,"mini_trainset_labeled.csv"), header=0)

# dataset = icmobi_dataloader.ICMOBIDatasetFormatter(df, transform=transform_pipeline)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

# cnt = 1
# for batch in dataloader:
#     print(f"Running batch {cnt}")
#     psd = batch['psd']       # (B, 1, 100)
#     ac = batch['ac']         # (B, 1, 100)
#     topo = batch['topo']     # (B, 1, 32, 32)
#     label = batch['label']   # (B, 7)
#     cnt += 1

#%% MAIN K-FOLD PROCESS
# -- Set random seeds for reproducibility
# Load dataset
# --- load data ---
rel_path = os.path.abspath("..\\data\\")
df = pd.read_csv(os.path.join(rel_path,"mini_trainset_labeled.csv"))
#(07/14/2025) JS, change this when the full .csv's are made

dataset = icmobi_dataloader.ICMOBIDatasetFormatter(df,
                                                   transform=transform_pipeline)

# Split into training and validation
val_size = int(len(dataset) * VALID_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#-- use distributed dataloading
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# train_loader = DataLoader(train_dataset,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True)
# val_loader = DataLoader(val_dataset,
#                         batch_size=BATCH_SIZE,
#                         shuffle=False)

#-- Load model
# model = icmobi_model.ICMoBiNetTrain()
rel_path = os.path.abspath("..\\icmobi_ext\\utils\\")
model = icmobi_model.ICMoBiNetTransferL(mat_path=os.path.join(rel_path,"netICL.mat"))
#(07/14/2025) JS, using transfer learning...

#-- Optionally freeze some layers (example: freeze all except the last conv and softmax)
# freezing_layers = "discriminator_conv"
# for name, param in model.named_parameters():
#     if freezing_layers not in name:
#         param.requires_grad = False
#(07/14/2025) JS, this is to avoid training the model with coefficients

#-- model to device    
# device = torch.device("cpu")
model.to(device)
model = DDP(model)  # No device_ids needed for CPU

#(07/14/2025) JS, need to include previous model coefficients into training here.