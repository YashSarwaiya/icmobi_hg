
# ...existing code...
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from icmobi_ext import icmobi_model
from icmobi_ext import icmobi_dataloader
import pandas as pd
import os

#%% CPU INITIALIZATION
def setup_distributed(backend="mpi"):
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

#--
rank, world_size, local_rank = setup_distributed("mpi")

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

#%% MAIN K-FOLD PROCESS
# -- Set random seeds for reproducibility

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
STEP_SIZE = 10
GAMMA = 0.5
VALID_SPLIT = 0.2

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

#-- Load model
# model = icmobi_model.ICMoBiNetTrain()
rel_path = os.path.abspath("..\\icmobi_ext\\utils\\")
model = icmobi_model.ICMoBiNetTransferL(mat_path=os.path.join(rel_path,"netICL.mat"))

#%% MOVE MODEL TO DEVICE

device = torch.device("cpu")
model.to(device)

model = DDP(model)  # No device_ids needed for CPU

#%%
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
STEP_SIZE = 10
GAMMA = 0.5
VALID_SPLIT = 0.2

def main_worker(rank, world_size):
    # Setup distributed
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500',
                            world_size=world_size, rank=rank)
    torch.manual_seed(0)

    #-- Load dataset
    rel_path = os.path.abspath("..\\data\\")
    df = pd.read_csv(os.path.join(rel_path,"mini_trainset_labeled.csv"))
    transform_pipeline = None
    dataset = icmobi_dataloader.ICMOBIDatasetFormatter(df, transform=transform_pipeline)

    val_size = int(len(dataset) * VALID_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #-- Distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    # Model
    rel_path_utils = os.path.abspath("..\\icmobi_ext\\utils\\")
    model = icmobi_model.ICMoBiNetTransferL(mat_path=os.path.join(rel_path_utils,"netICL.mat"))
    model = nn.parallel.DistributedDataParallel(model.to("cpu"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        print(f"\n[Rank {rank}] Epoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Training [Rank {rank}]", disable=(rank!=0)):
            images = batch['topo'].to("cpu")
            psds = batch['psd'].to("cpu")
            acs = batch['ac'].to("cpu")
            labels = batch['label'].to("cpu")
            optimizer.zero_grad()
            outputs = model(images, psds, acs)
            labels_cls = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels_cls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels_cls).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        if rank == 0:
            print(f"Train Loss: {train_loss / train_total:.4f}, Accuracy: {train_accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation [Rank {rank}]", disable=(rank!=0)):
                images = batch['topo'].to("cpu")
                psds = batch['psd'].to("cpu")
                acs = batch['ac'].to("cpu")
                labels = batch['label'].to("cpu")
                outputs = model(images, psds, acs)
                labels_cls = torch.argmax(labels, dim=1)
                loss = criterion(outputs, labels_cls)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels_cls).sum().item()
                val_total += labels.size(0)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        if rank == 0:
            print(f"Val Loss: {val_loss / val_total:.4f}, Accuracy: {val_accuracy:.4f}")

    if rank == 0:
        print("\nTraining complete.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = min(2, os.cpu_count())  # You can increase this if you want more processes
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
# ...existing code...