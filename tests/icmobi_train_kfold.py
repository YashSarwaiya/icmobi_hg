#%% IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
#--
from icmobi_ext import icmobi_model
from icmobi_ext import icmobi_dataloader
import pandas as pd
import os
# from icmobi_model import icldata, icmobi_model, plot_functions  # code adapted from https://github.com/lucapton/ICLabel-Dataset

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
# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
STEP_SIZE = 10
GAMMA = 0.5
VALID_SPLIT = 0.2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
# --- load data ---
rel_path = os.path.abspath("..\\data\\")
# df = pd.read_csv(os.path.join(rel_path,"mini_trainset_labeled.csv"))
df = pd.read_csv(os.path.join(rel_path,"trainset_labeled.csv"))
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
#(07/14/2025) JS, using transfer learning...

#-- Optionally freeze some layers (example: freeze all except the last conv and softmax)
# freezing_layers = "discriminator_conv"
# for name, param in model.named_parameters():
#     if freezing_layers not in name:
#         param.requires_grad = False
#(07/14/2025) JS, this is to avoid training the model with coefficients
    
model.to(device)
#(07/14/2025) JS, need to include previous model coefficients into training here.

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#-- Define optimizer (only parameters with requires_grad=True will be updated)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Training loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
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


# %%
