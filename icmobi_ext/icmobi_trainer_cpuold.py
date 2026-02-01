#%% IMPORTS
import torch
import torch.distributed as dist
import torch.nn as nn
# import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
# from torch.multiprocessing import Pool, set_start_method
# import torch.multiprocessing as mp
from tqdm import tqdm
#--
from icmobi_ext import icmobi_model
from icmobi_ext import icmobi_dataloader
import pandas as pd
import os
# import socket
# from icmobi_model import icldata, icmobi_model, plot_functions  # code adapted from https://github.com/lucapton/ICLabel-Dataset

#%% SETTING UP MULTIP
print(f"[Rank {os.environ.get('RANK', '?')}] "
      f"MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}, "
      f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

#%% CPU INITIALIZATION
def setup_distributed(backend="gloo"):
    """Initialize torch.distributed under SLURM or torchrun."""
    import os
    import socket
    import torch.distributed as dist

    # ---- detect mode ----
    # When launched via torchrun, these are provided automatically
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"Detected torchrun launch: rank={rank}, world_size={world_size}")
    # When launched via srun (without torchrun), use SLURM variables
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ.setdefault("MASTER_ADDR", socket.gethostname())
        os.environ.setdefault("MASTER_PORT", "12355")
        print(f"Detected SLURM launch: rank={rank}, world_size={world_size}")
    else:
        raise EnvironmentError("Distributed setup failed: no RANK or SLURM_PROCID found")

    # ---- init process group ----
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=world_size,
        rank=rank,
    )
    print(f"[Rank {rank}] Initialized process group (world size={world_size})")
    return rank, world_size

def cleanup():
    dist.destroy_process_group()
#enddef

#-- setup distributed computing for openmpi
rank, world_size = setup_distributed("gloo")
device = torch.device("cpu")

#%% MULTIPROCESSING/THREADING
print(torch.__config__.parallel_info())
#%% DATASET TRANSFORMS
transform_pipeline = transforms.Compose([
    icmobi_dataloader.RandomHorizontalFlipTopography(),
    icmobi_dataloader.RandomGaussianNoiseTopography(std=0.02),
    icmobi_dataloader.RandomTopographyDropout(max_size=6),
    icmobi_dataloader.NormalizeTopography(),
    
    icmobi_dataloader.AddNoisePSD(std=0.01),
    icmobi_dataloader.NormalizePSD(),

    icmobi_dataloader.AddNoiseAC(std=0.01),
    icmobi_dataloader.NormalizeAC()
])
transform_pipeline = None
#(07/14/2025) JS, torchvision doesn't work because of version 
# mismatch with PyTorch. I think its mainly because of the 
# Windows, but should to use version pairs: python setup.py install  # inside torchvision source dir
# or pip install torch==2.2.0 torchvision==0.17.0

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load dataset
# --- load data ---
rel_path = os.path.abspath("../data")
# df = pd.read_csv(os.path.join(rel_path,"mini_trainset_labeled.csv"))
df = pd.read_csv(os.path.join(rel_path,"trainset_labeled.csv"))
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

#-- Load model
# model = icmobi_model.ICMoBiNetTrain()
rel_path = os.path.abspath("../icmobi_ext/utils/")
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

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#-- Define optimizer (only parameters with requires_grad=True will be updated)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Training loop
for epoch in range(EPOCHS):
    train_sampler.set_epoch(epoch)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("-" * 20)    

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in tqdm(train_loader, desc="Training"):    
        images = batch['topo'].to(device)
        psds = batch['psd'].to(device)
        acs = batch['ac'].to(device)
        labels = batch['label'].to(device)
        #--
        print("images shape:", images.shape)
        print("psds shape:", psds.shape)
        print("acs shape:", acs.shape)
        #--
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

    train_accuracy = train_correct / train_total
    print(f"Train Loss: {train_loss / train_total:.4f}, Accuracy: {train_accuracy:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        # Wrap val_loader with tqdm for progress bar
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['topo'].to(device)
            psds = batch['psd'].to(device)
            acs = batch['ac'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images, psds, acs)
            labels_cls = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels_cls)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels_cls).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    print(f"Val Loss: {val_loss / val_total:.4f}, Accuracy: {val_accuracy:.4f}")
    
print("\nTraining complete.")
cleanup()