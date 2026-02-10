# # training using just old data
# # #%% IMPORTS
# # import torch
# # import torch.distributed as dist
# # import torch.nn as nn
# # from torch.nn.parallel import DistributedDataParallel as DDP
# # from torch.utils.data import DataLoader, random_split
# # from torch.utils.data.distributed import DistributedSampler
# # from torchvision import transforms
# # from torch.optim.lr_scheduler import StepLR
# # from tqdm import tqdm
# # #--
# # from icmobi_ext import icmobi_model
# # from icmobi_ext import icmobi_dataloader
# # import pandas as pd
# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import torch.onnx

# # #%% SETTING UP MULTIP
# # print(f"[Rank {os.environ.get('RANK', '?')}] "
# #       f"MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}, "
# #       f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

# # #%% CPU INITIALIZATION
# # def setup_distributed(backend="gloo"):
# #     """Initialize torch.distributed under SLURM or torchrun."""
# #     import os
# #     import socket
# #     import torch.distributed as dist

# #     # ---- detect mode ----
# #     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
# #         rank = int(os.environ["RANK"])
# #         world_size = int(os.environ["WORLD_SIZE"])
# #         print(f"Detected torchrun launch: rank={rank}, world_size={world_size}")
# #     elif "SLURM_PROCID" in os.environ:
# #         rank = int(os.environ["SLURM_PROCID"])
# #         world_size = int(os.environ["SLURM_NTASKS"])
# #         os.environ["RANK"] = str(rank)
# #         os.environ["WORLD_SIZE"] = str(world_size)
# #         os.environ.setdefault("MASTER_ADDR", socket.gethostname())
# #         os.environ.setdefault("MASTER_PORT", "12355")
# #         print(f"Detected SLURM launch: rank={rank}, world_size={world_size}")
# #     else:
# #         raise EnvironmentError("Distributed setup failed: no RANK or SLURM_PROCID found")

# #     # ---- init process group ----
# #     dist.init_process_group(
# #         backend=backend,
# #         init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
# #         world_size=world_size,
# #         rank=rank,
# #     )
# #     print(f"[Rank {rank}] Initialized process group (world size={world_size})")
# #     return rank, world_size

# # def cleanup():
# #     dist.destroy_process_group()

# # #-- setup distributed computing for openmpi
# # rank, world_size = setup_distributed("gloo")
# # device = torch.device("cpu")

# # #%% MULTIPROCESSING/THREADING
# # print(torch.__config__.parallel_info())

# # #%% DATASET TRANSFORMS
# # transform_pipeline = transforms.Compose([
# #     icmobi_dataloader.RandomHorizontalFlipTopography(),
# #     icmobi_dataloader.RandomGaussianNoiseTopography(std=0.02),
# #     icmobi_dataloader.RandomTopographyDropout(max_size=6),
# #     icmobi_dataloader.NormalizeTopography(),
    
# #     icmobi_dataloader.AddNoisePSD(std=0.01),
# #     icmobi_dataloader.NormalizePSD(),

# #     icmobi_dataloader.AddNoiseAC(std=0.01),
# #     icmobi_dataloader.NormalizeAC()
# # ])
# # transform_pipeline = None

# # #%% EVALUATION FUNCTIONS (PyTorch Native)
# # def compute_confusion_matrix(y_true, y_pred, num_classes):
# #     """Compute confusion matrix using PyTorch"""
# #     cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
# #     for t, p in zip(y_true, y_pred):
# #         cm[t, p] += 1
# #     return cm

# # def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
# #     """Plot and save confusion matrix using PyTorch"""
# #     num_classes = len(class_names)
# #     cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    
# #     plt.figure(figsize=(10, 8))
# #     sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues', 
# #                 xticklabels=class_names, yticklabels=class_names)
# #     plt.title('Confusion Matrix')
# #     plt.ylabel('True Label')
# #     plt.xlabel('Predicted Label')
# #     plt.tight_layout()
# #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     plt.close()
# #     print(f"Confusion matrix saved to {save_path}")
# #     return cm

# # def compute_roc_curve(y_true, y_scores, pos_label):
# #     """Compute ROC curve using PyTorch"""
# #     # Sort by score
# #     sorted_indices = torch.argsort(y_scores, descending=True)
# #     y_true_sorted = y_true[sorted_indices]
# #     y_scores_sorted = y_scores[sorted_indices]
    
# #     # Compute TPR and FPR at each threshold
# #     tps = torch.cumsum(y_true_sorted, dim=0)
# #     fps = torch.cumsum(1 - y_true_sorted, dim=0)
    
# #     total_pos = tps[-1]
# #     total_neg = fps[-1]
    
# #     tpr = tps.float() / total_pos.float() if total_pos > 0 else torch.zeros_like(tps).float()
# #     fpr = fps.float() / total_neg.float() if total_neg > 0 else torch.zeros_like(fps).float()
    
# #     # Add (0,0) and (1,1) points
# #     fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])
# #     tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
    
# #     return fpr, tpr

# # def compute_auc(fpr, tpr):
# #     """Compute AUC using trapezoidal rule"""
# #     # Sort by fpr
# #     sorted_indices = torch.argsort(fpr)
# #     fpr_sorted = fpr[sorted_indices]
# #     tpr_sorted = tpr[sorted_indices]
    
# #     # Trapezoidal integration
# #     auc = torch.trapz(tpr_sorted, fpr_sorted)
# #     return auc.item()

# # def plot_roc_curves(y_true, y_probs, class_names, save_path='roc_curves.png'):
# #     """Plot ROC curves for multi-class classification using PyTorch"""
# #     num_classes = len(class_names)
    
# #     # Convert to one-hot
# #     y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=num_classes)
    
# #     fpr_dict = {}
# #     tpr_dict = {}
# #     auc_dict = {}
    
# #     # Compute ROC for each class
# #     for i in range(num_classes):
# #         fpr, tpr = compute_roc_curve(y_true_onehot[:, i], y_probs[:, i], pos_label=1)
# #         fpr_dict[i] = fpr
# #         tpr_dict[i] = tpr
# #         auc_dict[i] = compute_auc(fpr, tpr)
    
# #     # Compute micro-average
# #     y_true_flat = y_true_onehot.reshape(-1)
# #     y_probs_flat = y_probs.reshape(-1)
# #     fpr_micro, tpr_micro = compute_roc_curve(y_true_flat, y_probs_flat, pos_label=1)
# #     auc_dict['micro'] = compute_auc(fpr_micro, tpr_micro)
    
# #     # Plot
# #     plt.figure(figsize=(10, 8))
# #     colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
# #     for i, color in zip(range(num_classes), colors):
# #         plt.plot(fpr_dict[i].numpy(), tpr_dict[i].numpy(), color=color, lw=2,
# #                 label=f'{class_names[i]} (AUC = {auc_dict[i]:.3f})')
    
# #     plt.plot(fpr_micro.numpy(), tpr_micro.numpy(), color='deeppink', linestyle='--', lw=2,
# #             label=f'Micro-average (AUC = {auc_dict["micro"]:.3f})')
    
# #     plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
# #     plt.xlim([0.0, 1.0])
# #     plt.ylim([0.0, 1.05])
# #     plt.xlabel('False Positive Rate')
# #     plt.ylabel('True Positive Rate')
# #     plt.title('ROC Curves - Multi-Class Classification')
# #     plt.legend(loc="lower right")
# #     plt.grid(alpha=0.3)
# #     plt.tight_layout()
# #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     plt.close()
# #     print(f"ROC curves saved to {save_path}")
    
# #     return auc_dict

# # def compute_classification_metrics(y_true, y_pred, num_classes):
# #     """Compute precision, recall, F1 for each class using PyTorch"""
# #     metrics = {}
    
# #     for i in range(num_classes):
# #         # Binary masks for class i
# #         true_pos = ((y_true == i) & (y_pred == i)).sum().float()
# #         false_pos = ((y_true != i) & (y_pred == i)).sum().float()
# #         false_neg = ((y_true == i) & (y_pred != i)).sum().float()
# #         true_neg = ((y_true != i) & (y_pred != i)).sum().float()
        
# #         # Precision, Recall, F1
# #         precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else torch.tensor(0.0)
# #         recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else torch.tensor(0.0)
# #         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
        
# #         support = (y_true == i).sum()
        
# #         metrics[i] = {
# #             'precision': precision.item(),
# #             'recall': recall.item(),
# #             'f1': f1.item(),
# #             'support': support.item()
# #         }
    
# #     # Compute macro and weighted averages
# #     precisions = [metrics[i]['precision'] for i in range(num_classes)]
# #     recalls = [metrics[i]['recall'] for i in range(num_classes)]
# #     f1s = [metrics[i]['f1'] for i in range(num_classes)]
# #     supports = [metrics[i]['support'] for i in range(num_classes)]
# #     total_support = sum(supports)
    
# #     metrics['macro_avg'] = {
# #         'precision': np.mean(precisions),
# #         'recall': np.mean(recalls),
# #         'f1': np.mean(f1s),
# #         'support': total_support
# #     }
    
# #     metrics['weighted_avg'] = {
# #         'precision': np.average(precisions, weights=supports),
# #         'recall': np.average(recalls, weights=supports),
# #         'f1': np.average(f1s, weights=supports),
# #         'support': total_support
# #     }
    
# #     return metrics

# # def print_classification_report(metrics, class_names):
# #     """Print classification report similar to sklearn"""
# #     print("\nClassification Report:")
# #     print("-" * 70)
# #     print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
# #     print("-" * 70)
    
# #     for i, class_name in enumerate(class_names):
# #         m = metrics[i]
# #         print(f"{class_name:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
    
# #     print("-" * 70)
# #     m = metrics['macro_avg']
# #     print(f"{'Macro avg':<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
    
# #     m = metrics['weighted_avg']
# #     print(f"{'Weighted avg':<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
# #     print("-" * 70)

# # def save_model_diagnostics(train_history, val_history, save_path='training_history.png'):
# #     """Plot and save training diagnostics"""
# #     fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
# #     # Loss plot
# #     axes[0].plot(train_history['loss'], label='Train Loss', marker='o')
# #     axes[0].plot(val_history['loss'], label='Val Loss', marker='s')
# #     axes[0].set_xlabel('Epoch')
# #     axes[0].set_ylabel('Loss')
# #     axes[0].set_title('Training and Validation Loss')
# #     axes[0].legend()
# #     axes[0].grid(alpha=0.3)
    
# #     # Accuracy plot
# #     axes[1].plot(train_history['accuracy'], label='Train Accuracy', marker='o')
# #     axes[1].plot(val_history['accuracy'], label='Val Accuracy', marker='s')
# #     axes[1].set_xlabel('Epoch')
# #     axes[1].set_ylabel('Accuracy')
# #     axes[1].set_title('Training and Validation Accuracy')
# #     axes[1].legend()
# #     axes[1].grid(alpha=0.3)
    
# #     plt.tight_layout()
# #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     plt.close()
# #     print(f"Training history saved to {save_path}")

# # def evaluate_model(model, data_loader, device, class_names):
# #     """Comprehensive model evaluation using PyTorch"""
# #     model.eval()
# #     all_preds = []
# #     all_labels = []
# #     all_probs = []
    
# #     with torch.no_grad():
# #         for batch in tqdm(data_loader, desc="Evaluating"):
# #             images = batch['topo'].to(device)
# #             psds = batch['psd'].to(device)
# #             acs = batch['ac'].to(device)
# #             labels = batch['label'].to(device)
            
# #             outputs = model(images, psds, acs)
# #             probs = torch.softmax(outputs, dim=1)
# #             _, predicted = torch.max(outputs, 1)
# #             labels_cls = torch.argmax(labels, dim=1)
            
# #             all_preds.append(predicted.cpu())
# #             all_labels.append(labels_cls.cpu())
# #             all_probs.append(probs.cpu())
    
# #     all_preds = torch.cat(all_preds)
# #     all_labels = torch.cat(all_labels)
# #     all_probs = torch.cat(all_probs)
    
# #     # Compute and print classification metrics
# #     num_classes = len(class_names)
# #     metrics = compute_classification_metrics(all_labels, all_preds, num_classes)
# #     print_classification_report(metrics, class_names)
    
# #     return all_labels, all_preds, all_probs, metrics

# # def export_to_onnx(model, save_path, device, input_shapes):
# #     """Export model to ONNX format"""
# #     model.eval()
    
# #     # Create dummy inputs
# #     dummy_topo = torch.randn(1, *input_shapes['topo']).to(device)
# #     dummy_psd = torch.randn(1, *input_shapes['psd']).to(device)
# #     dummy_ac = torch.randn(1, *input_shapes['ac']).to(device)
    
# #     # Export
# #     torch.onnx.export(
# #         model.module if isinstance(model, DDP) else model,
# #         (dummy_topo, dummy_psd, dummy_ac),
# #         save_path,
# #         export_params=True,
# #         opset_version=11,
# #         do_constant_folding=True,
# #         input_names=['topography', 'psd', 'autocorrelation'],
# #         output_names=['output'],
# #         dynamic_axes={
# #             'topography': {0: 'batch_size'},
# #             'psd': {0: 'batch_size'},
# #             'autocorrelation': {0: 'batch_size'},
# #             'output': {0: 'batch_size'}
# #         }
# #     )
# #     print(f"Model exported to ONNX: {save_path}")

# # #%% MAIN K-FOLD PROCESS
# # # Hyperparameters
# # BATCH_SIZE = 32
# # EPOCHS = 30
# # LEARNING_RATE = 1e-4
# # STEP_SIZE = 10
# # GAMMA = 0.5
# # VALID_SPLIT = 0.2

# # # Device configuration
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(device)

# # # Load dataset
# # rel_path = os.path.abspath("../data")
# # df = pd.read_csv(os.path.join(rel_path,"trainset_labeled.csv"))

# # dataset = icmobi_dataloader.ICMOBIDatasetFormatter(df, transform=transform_pipeline)

# # # Split into training and validation
# # val_size = int(len(dataset) * VALID_SPLIT)
# # train_size = len(dataset) - val_size
# # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # # Use distributed dataloading
# # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
# # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

# # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
# # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# # # Load model
# # rel_path = os.path.abspath("../icmobi_ext/utils/")
# # model = icmobi_model.ICMoBiNetTransferL(mat_path=os.path.join(rel_path,"netICL.mat"))

# # # Model to device    
# # model.to(device)
# # model = DDP(model)

# # # Loss and optimizer
# # criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
# # scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# # # Training history tracking
# # train_history = {'loss': [], 'accuracy': []}
# # val_history = {'loss': [], 'accuracy': []}

# # # Define class names (adjust based on your dataset)
# # class_names = ['Brain', 'Muscle', 'Eye', 'Heart', 'Line_Noise', 'Channel_Noise', 'Other']

# # # Training loop
# # for epoch in range(EPOCHS):
# #     train_sampler.set_epoch(epoch)
# #     print(f"\nEpoch {epoch+1}/{EPOCHS}")
# #     print("-" * 20)    

# #     model.train()
# #     train_loss = 0.0
# #     train_correct = 0
# #     train_total = 0

# #     for batch in tqdm(train_loader, desc="Training"):    
# #         images = batch['topo'].to(device)
# #         psds = batch['psd'].to(device)
# #         acs = batch['ac'].to(device)
# #         labels = batch['label'].to(device)
        
# #         optimizer.zero_grad()
# #         outputs = model(images, psds, acs)
# #         labels_cls = torch.argmax(labels, dim=1)
# #         loss = criterion(outputs, labels_cls)
# #         loss.backward()
# #         optimizer.step()

# #         train_loss += loss.item() * images.size(0)
# #         _, predicted = torch.max(outputs, 1)
# #         train_correct += (predicted == labels_cls).sum().item()
# #         train_total += labels.size(0)

# #     scheduler.step()

# #     train_accuracy = train_correct / train_total
# #     train_epoch_loss = train_loss / train_total
# #     train_history['loss'].append(train_epoch_loss)
# #     train_history['accuracy'].append(train_accuracy)
# #     print(f"Train Loss: {train_epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}")

# #     # Validation
# #     model.eval()
# #     val_loss = 0.0
# #     val_correct = 0
# #     val_total = 0

# #     with torch.no_grad():
# #         for batch in tqdm(val_loader, desc="Validation"):
# #             images = batch['topo'].to(device)
# #             psds = batch['psd'].to(device)
# #             acs = batch['ac'].to(device)
# #             labels = batch['label'].to(device)
# #             outputs = model(images, psds, acs)
# #             labels_cls = torch.argmax(labels, dim=1)
# #             loss = criterion(outputs, labels_cls)

# #             val_loss += loss.item() * images.size(0)
# #             _, predicted = torch.max(outputs, 1)
# #             val_correct += (predicted == labels_cls).sum().item()
# #             val_total += labels.size(0)

# #     val_accuracy = val_correct / val_total
# #     val_epoch_loss = val_loss / val_total
# #     val_history['loss'].append(val_epoch_loss)
# #     val_history['accuracy'].append(val_accuracy)
# #     print(f"Val Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")

# # print("\nTraining complete.")

# # #%% EVALUATION AND EXPORT (Only on rank 0)
# # if rank == 0:
# #     print("\n" + "="*50)
# #     print("FINAL MODEL EVALUATION")
# #     print("="*50)
    
# #     # Create output directory
# #     output_dir = "../outputs"
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     # Evaluate on validation set
# #     y_true, y_pred, y_probs, metrics = evaluate_model(model, val_loader, device, class_names)
    
# #     # Generate confusion matrix
# #     cm = plot_confusion_matrix(y_true, y_pred, class_names, 
# #                                save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
# #     # Generate ROC curves
# #     auc_scores = plot_roc_curves(y_true, y_probs, class_names,
# #                                  save_path=os.path.join(output_dir, 'roc_curves.png'))
    
# #     # Save training history
# #     save_model_diagnostics(train_history, val_history,
# #                           save_path=os.path.join(output_dir, 'training_history.png'))
    
# #     # Save metrics to CSV
# #     metrics_df = pd.DataFrame({
# #         'Epoch': list(range(1, EPOCHS + 1)),
# #         'Train_Loss': train_history['loss'],
# #         'Train_Accuracy': train_history['accuracy'],
# #         'Val_Loss': val_history['loss'],
# #         'Val_Accuracy': val_history['accuracy']
# #     })
# #     metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)
# #     print(f"Training metrics saved to {os.path.join(output_dir, 'training_metrics.csv')}")
    
# #     # Save classification metrics
# #     class_metrics_data = []
# #     for i, class_name in enumerate(class_names):
# #         class_metrics_data.append({
# #             'Class': class_name,
# #             'Precision': metrics[i]['precision'],
# #             'Recall': metrics[i]['recall'],
# #             'F1-Score': metrics[i]['f1'],
# #             'Support': int(metrics[i]['support'])
# #         })
# #     class_metrics_df = pd.DataFrame(class_metrics_data)
# #     class_metrics_df.to_csv(os.path.join(output_dir, 'classification_metrics.csv'), index=False)
# #     print(f"Classification metrics saved to {os.path.join(output_dir, 'classification_metrics.csv')}")
    
# #     # Save AUC scores
# #     auc_df = pd.DataFrame({
# #         'Class': class_names + ['Micro-average'],
# #         'AUC': [auc_scores[i] for i in range(len(class_names))] + [auc_scores['micro']]
# #     })
# #     auc_df.to_csv(os.path.join(output_dir, 'auc_scores.csv'), index=False)
# #     print(f"AUC scores saved to {os.path.join(output_dir, 'auc_scores.csv')}")
    
# #     # Export to ONNX
# #     # Get input shapes from first batch
# #     sample_batch = next(iter(val_loader))
# #     input_shapes = {
# #         'topo': sample_batch['topo'].shape[1:],
# #         'psd': sample_batch['psd'].shape[1:],
# #         'ac': sample_batch['ac'].shape[1:]
# #     }
    
# #     export_to_onnx(model, 
# #                    os.path.join(output_dir, 'icmobi_model.onnx'),
# #                    device,
# #                    input_shapes)
    
# #     # Save PyTorch model weights
# #     torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
# #                os.path.join(output_dir, 'icmobi_model_weights.pth'))
# #     print(f"PyTorch weights saved to {os.path.join(output_dir, 'icmobi_model_weights.pth')}")
    
# #     print("\n" + "="*50)
# #     print("All evaluation outputs saved to:", output_dir)
# #     print("="*50)

# # cleanup()























# #tranining using both and new both data and and some imporvment like early stop 
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, Subset
# from torch.utils.data.distributed import DistributedSampler
# from torchvision import transforms
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# import copy
# #--
# from icmobi_ext import icmobi_model
# from icmobi_ext import icmobi_dataloader
# import pandas as pd
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch.onnx

# #%% SETTING UP MULTIP
# print(f"[Rank {os.environ.get('RANK', '?')}] "
#       f"MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}, "
#       f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

# #%% CPU INITIALIZATION
# def setup_distributed(backend="gloo"):
#     """Initialize torch.distributed under SLURM or torchrun."""
#     import os
#     import socket
#     import torch.distributed as dist

#     # ---- detect mode ----
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#         print(f"Detected torchrun launch: rank={rank}, world_size={world_size}")
#     elif "SLURM_PROCID" in os.environ:
#         rank = int(os.environ["SLURM_PROCID"])
#         world_size = int(os.environ["SLURM_NTASKS"])
#         os.environ["RANK"] = str(rank)
#         os.environ["WORLD_SIZE"] = str(world_size)
#         os.environ.setdefault("MASTER_ADDR", socket.gethostname())
#         os.environ.setdefault("MASTER_PORT", "12355")
#         print(f"Detected SLURM launch: rank={rank}, world_size={world_size}")
#     else:
#         raise EnvironmentError("Distributed setup failed: no RANK or SLURM_PROCID found")

#     # ---- init process group ----
#     dist.init_process_group(
#         backend=backend,
#         init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
#         world_size=world_size,
#         rank=rank,
#     )
#     print(f"[Rank {rank}] Initialized process group (world size={world_size})")
#     return rank, world_size

# def cleanup():
#     dist.destroy_process_group()

# #-- setup distributed computing for openmpi
# rank, world_size = setup_distributed("gloo")
# device = torch.device("cpu")

# #%% MULTIPROCESSING/THREADING
# print(torch.__config__.parallel_info())

# #%% DATASET TRANSFORMS
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
# transform_pipeline = None

# #%% EVALUATION FUNCTIONS (PyTorch Native)
# def compute_confusion_matrix(y_true, y_pred, num_classes):
#     """Compute confusion matrix using PyTorch"""
#     cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
#     for t, p in zip(y_true, y_pred):
#         cm[t, p] += 1
#     return cm

# def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
#     """Plot and save confusion matrix using PyTorch"""
#     num_classes = len(class_names)
#     cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Confusion matrix saved to {save_path}")
#     return cm

# def compute_roc_curve(y_true, y_scores, pos_label):
#     """Compute ROC curve using PyTorch"""
#     # Sort by score
#     sorted_indices = torch.argsort(y_scores, descending=True)
#     y_true_sorted = y_true[sorted_indices]
#     y_scores_sorted = y_scores[sorted_indices]
    
#     # Compute TPR and FPR at each threshold
#     tps = torch.cumsum(y_true_sorted, dim=0)
#     fps = torch.cumsum(1 - y_true_sorted, dim=0)
    
#     total_pos = tps[-1]
#     total_neg = fps[-1]
    
#     tpr = tps.float() / total_pos.float() if total_pos > 0 else torch.zeros_like(tps).float()
#     fpr = fps.float() / total_neg.float() if total_neg > 0 else torch.zeros_like(fps).float()
    
#     # Add (0,0) and (1,1) points
#     fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])
#     tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
    
#     return fpr, tpr

# def compute_auc(fpr, tpr):
#     """Compute AUC using trapezoidal rule"""
#     # Sort by fpr
#     sorted_indices = torch.argsort(fpr)
#     fpr_sorted = fpr[sorted_indices]
#     tpr_sorted = tpr[sorted_indices]
    
#     # Trapezoidal integration
#     auc = torch.trapz(tpr_sorted, fpr_sorted)
#     return auc.item()

# def plot_roc_curves(y_true, y_probs, class_names, save_path='roc_curves.png'):
#     """Plot ROC curves for multi-class classification using PyTorch"""
#     num_classes = len(class_names)
    
#     # Convert to one-hot
#     y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=num_classes)
    
#     fpr_dict = {}
#     tpr_dict = {}
#     auc_dict = {}
    
#     # Compute ROC for each class
#     for i in range(num_classes):
#         fpr, tpr = compute_roc_curve(y_true_onehot[:, i], y_probs[:, i], pos_label=1)
#         fpr_dict[i] = fpr
#         tpr_dict[i] = tpr
#         auc_dict[i] = compute_auc(fpr, tpr)
    
#     # Compute micro-average
#     y_true_flat = y_true_onehot.reshape(-1)
#     y_probs_flat = y_probs.reshape(-1)
#     fpr_micro, tpr_micro = compute_roc_curve(y_true_flat, y_probs_flat, pos_label=1)
#     auc_dict['micro'] = compute_auc(fpr_micro, tpr_micro)
    
#     # Plot
#     plt.figure(figsize=(10, 8))
#     colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
#     for i, color in zip(range(num_classes), colors):
#         plt.plot(fpr_dict[i].numpy(), tpr_dict[i].numpy(), color=color, lw=2,
#                 label=f'{class_names[i]} (AUC = {auc_dict[i]:.3f})')
    
#     plt.plot(fpr_micro.numpy(), tpr_micro.numpy(), color='deeppink', linestyle='--', lw=2,
#             label=f'Micro-average (AUC = {auc_dict["micro"]:.3f})')
    
#     plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curves - Multi-Class Classification')
#     plt.legend(loc="lower right")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"ROC curves saved to {save_path}")
    
#     return auc_dict

# def compute_classification_metrics(y_true, y_pred, num_classes):
#     """Compute precision, recall, F1 for each class using PyTorch"""
#     metrics = {}
    
#     for i in range(num_classes):
#         # Binary masks for class i
#         true_pos = ((y_true == i) & (y_pred == i)).sum().float()
#         false_pos = ((y_true != i) & (y_pred == i)).sum().float()
#         false_neg = ((y_true == i) & (y_pred != i)).sum().float()
#         true_neg = ((y_true != i) & (y_pred != i)).sum().float()
        
#         # Precision, Recall, F1
#         precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else torch.tensor(0.0)
#         recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else torch.tensor(0.0)
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
        
#         support = (y_true == i).sum()
        
#         metrics[i] = {
#             'precision': precision.item(),
#             'recall': recall.item(),
#             'f1': f1.item(),
#             'support': support.item()
#         }
    
#     # Compute macro and weighted averages
#     precisions = [metrics[i]['precision'] for i in range(num_classes)]
#     recalls = [metrics[i]['recall'] for i in range(num_classes)]
#     f1s = [metrics[i]['f1'] for i in range(num_classes)]
#     supports = [metrics[i]['support'] for i in range(num_classes)]
#     total_support = sum(supports)
    
#     metrics['macro_avg'] = {
#         'precision': np.mean(precisions),
#         'recall': np.mean(recalls),
#         'f1': np.mean(f1s),
#         'support': total_support
#     }
    
#     metrics['weighted_avg'] = {
#         'precision': np.average(precisions, weights=supports) if total_support > 0 else 0,
#         'recall': np.average(recalls, weights=supports) if total_support > 0 else 0,
#         'f1': np.average(f1s, weights=supports) if total_support > 0 else 0,
#         'support': total_support
#     }
    
#     return metrics

# def print_classification_report(metrics, class_names):
#     """Print classification report similar to sklearn"""
#     print("\nClassification Report:")
#     print("-" * 70)
#     print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
#     print("-" * 70)
    
#     for i, class_name in enumerate(class_names):
#         m = metrics[i]
#         print(f"{class_name:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
    
#     print("-" * 70)
#     m = metrics['macro_avg']
#     print(f"{'Macro avg':<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
    
#     m = metrics['weighted_avg']
#     print(f"{'Weighted avg':<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
#     print("-" * 70)

# def save_model_diagnostics(train_history, val_history, best_epoch=None, save_path='training_history.png'):
#     """Plot and save training diagnostics"""
#     fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
#     # Loss plot
#     axes[0].plot(train_history['loss'], label='Train Loss', marker='o')
#     axes[0].plot(val_history['loss'], label='Val Loss', marker='s')
#     if best_epoch is not None:
#         axes[0].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Loss')
#     axes[0].set_title('Training and Validation Loss')
#     axes[0].legend()
#     axes[0].grid(alpha=0.3)
    
#     # Accuracy plot
#     axes[1].plot(train_history['accuracy'], label='Train Accuracy', marker='o')
#     axes[1].plot(val_history['accuracy'], label='Val Accuracy', marker='s')
#     if best_epoch is not None:
#         axes[1].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Accuracy')
#     axes[1].set_title('Training and Validation Accuracy')
#     axes[1].legend()
#     axes[1].grid(alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Training history saved to {save_path}")

# def evaluate_model(model, data_loader, device, class_names):
#     """Comprehensive model evaluation using PyTorch"""
#     model.eval()
#     all_preds = []
#     all_labels = []
#     all_probs = []
    
#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc="Evaluating"):
#             images = batch['topo'].to(device)
#             psds = batch['psd'].to(device)
#             acs = batch['ac'].to(device)
#             labels = batch['label'].to(device)
            
#             outputs = model(images, psds, acs)
#             probs = torch.softmax(outputs, dim=1)
#             _, predicted = torch.max(outputs, 1)
#             labels_cls = torch.argmax(labels, dim=1)
            
#             all_preds.append(predicted.cpu())
#             all_labels.append(labels_cls.cpu())
#             all_probs.append(probs.cpu())
    
#     all_preds = torch.cat(all_preds)
#     all_labels = torch.cat(all_labels)
#     all_probs = torch.cat(all_probs)
    
#     # Compute and print classification metrics
#     num_classes = len(class_names)
#     metrics = compute_classification_metrics(all_labels, all_preds, num_classes)
#     print_classification_report(metrics, class_names)
    
#     return all_labels, all_preds, all_probs, metrics

# def export_to_onnx(model, save_path, device, input_shapes):
#     """Export model to ONNX format"""
#     model.eval()
    
#     # Create dummy inputs
#     dummy_topo = torch.randn(1, *input_shapes['topo']).to(device)
#     dummy_psd = torch.randn(1, *input_shapes['psd']).to(device)
#     dummy_ac = torch.randn(1, *input_shapes['ac']).to(device)
    
#     # Export
#     torch.onnx.export(
#         model.module if isinstance(model, DDP) else model,
#         (dummy_topo, dummy_psd, dummy_ac),
#         save_path,
#         export_params=True,
#         opset_version=11,
#         do_constant_folding=True,
#         input_names=['topography', 'psd', 'autocorrelation'],
#         output_names=['output'],
#         dynamic_axes={
#             'topography': {0: 'batch_size'},
#             'psd': {0: 'batch_size'},
#             'autocorrelation': {0: 'batch_size'},
#             'output': {0: 'batch_size'}
#         }
#     )
#     print(f"Model exported to ONNX: {save_path}")

# #%% ============================================================
# # CONFIGURATION
# # ============================================================
# # Hyperparameters
# BATCH_SIZE = 32
# EPOCHS = 20                # Reduced from 30 (early stopping will handle it)
# LEARNING_RATE = 1e-4
# VALID_SPLIT = 0.2
# RANDOM_SEED = 42           # Fixed seed for reproducibility

# # Early Stopping Configuration
# PATIENCE = 10              # Stop if no improvement for 10 epochs
# MIN_DELTA = 0.001          # Minimum improvement to count

# # Class names (order matches LABEL_1 through LABEL_7)
# class_names = ['Brain', 'Muscle', 'Eye', 'Heart', 'Line_Noise', 'Channel_Noise', 'Other']

# # Label columns in the CSV
# LABEL_COLS = ['LABEL_1', 'LABEL_2', 'LABEL_3', 'LABEL_4', 'LABEL_5', 'LABEL_6', 'LABEL_7']

# #%% ============================================================
# # LOAD AND COMBINE DATASETS
# # ============================================================
# print("\n" + "="*50)
# print("LOADING DATASETS")
# print("="*50)

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")

# # Load both datasets
# rel_path = os.path.abspath("../data")

# df1 = pd.read_csv(os.path.join(rel_path, "trainset_labeled.csv"))
# print(f"Dataset 1: {len(df1)} samples")

# df2 = pd.read_csv(os.path.join(rel_path, "trainset_labeled_2.csv"))
# print(f"Dataset 2: {len(df2)} samples")

# # Combine datasets
# df = pd.concat([df1, df2], ignore_index=True)
# print(f"Combined dataset: {len(df)} samples")

# #%% ============================================================
# # CREATE ic_label FROM LABEL_1-7 (FIXED!)
# # ============================================================
# # Convert soft labels (LABEL_1-7) to hard labels (ic_label)
# # ic_label = name of the class with highest probability

# print("\nConverting soft labels to hard labels...")
# label_matrix = df[LABEL_COLS].values  # Shape: (N, 7)
# hard_label_indices = np.argmax(label_matrix, axis=1)  # Get index of max probability
# df['ic_label'] = [class_names[i] for i in hard_label_indices]  # Map to class name

# # Print class distribution
# print("\nClass Distribution:")
# print("-" * 40)
# class_counts = df['ic_label'].value_counts()
# for class_name in class_names:
#     count = class_counts.get(class_name, 0)
#     print(f"  {class_name:<15}: {count:5} samples ({100*count/len(df):5.1f}%)")
# print("-" * 40)

# #%% ============================================================
# # STRATIFIED SPLIT WITH FIXED SEED
# # ============================================================
# print("\n" + "="*50)
# print("CREATING STRATIFIED TRAIN/VAL SPLIT")
# print("="*50)

# # Get labels for stratification
# labels = df['ic_label'].values

# # Create indices
# indices = np.arange(len(df))

# # Stratified split with fixed seed
# train_idx, val_idx = train_test_split(
#     indices,
#     test_size=VALID_SPLIT,
#     random_state=RANDOM_SEED,    # Fixed seed for reproducibility
#     stratify=labels               # Maintains class proportions
# )

# print(f"Training samples: {len(train_idx)} ({100*len(train_idx)/len(df):.1f}%)")
# print(f"Validation samples: {len(val_idx)} ({100*len(val_idx)/len(df):.1f}%)")

# # Verify stratification worked
# print("\nVerifying stratification (should be ~same proportions):")
# train_labels = labels[train_idx]
# val_labels = labels[val_idx]
# for class_name in class_names:
#     train_pct = 100 * np.sum(train_labels == class_name) / len(train_labels)
#     val_pct = 100 * np.sum(val_labels == class_name) / len(val_labels)
#     print(f"  {class_name:<15}: Train {train_pct:5.1f}% | Val {val_pct:5.1f}%")

# #%% ============================================================
# # CREATE DATASETS AND DATALOADERS
# # ============================================================
# # Create full dataset
# dataset = icmobi_dataloader.ICMOBIDatasetFormatter(df, transform=transform_pipeline)

# # Create subset datasets using stratified indices
# train_dataset = Subset(dataset, train_idx)
# val_dataset = Subset(dataset, val_idx)

# # Use distributed dataloading
# train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
# val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# #%% ============================================================
# # COMPUTE CLASS WEIGHTS FOR IMBALANCED DATA
# # ============================================================
# print("\n" + "="*50)
# print("COMPUTING CLASS WEIGHTS")
# print("="*50)

# # Count samples per class in training set
# train_class_counts = []
# for class_name in class_names:
#     count = np.sum(train_labels == class_name)
#     train_class_counts.append(max(count, 1))  # Avoid division by zero
    
# train_class_counts = np.array(train_class_counts)

# # Compute weights (inverse frequency, normalized)
# class_weights = 1.0 / train_class_counts
# class_weights = class_weights / class_weights.sum() * len(class_names)
# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# print("Class weights (higher = more emphasis on this class):")
# for i, class_name in enumerate(class_names):
#     print(f"  {class_name:<15}: {class_weights[i].item():.3f} (samples: {train_class_counts[i]})")

# #%% ============================================================
# # LOAD MODEL
# # ============================================================
# print("\n" + "="*50)
# print("LOADING MODEL")
# print("="*50)

# rel_path = os.path.abspath("../icmobi_ext/utils/")
# model = icmobi_model.ICMoBiNetTransferL(mat_path=os.path.join(rel_path, "netICL.mat"))

# # Model to device    
# model.to(device)
# model = DDP(model)

# print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# #%% ============================================================
# # LOSS, OPTIMIZER, SCHEDULER
# # ============================================================
# # Loss with class weights (handles imbalanced data)
# criterion = nn.CrossEntropyLoss(weight=class_weights)

# # Optimizer
# optimizer = torch.optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()), 
#     lr=LEARNING_RATE
# )

# # Learning rate scheduler (reduces LR when validation plateaus)
# scheduler = ReduceLROnPlateau(
#     optimizer, 
#     mode='max',           # Maximize validation accuracy
#     factor=0.5,           # Reduce LR by half
#     patience=5           # Wait 5 epochs before reducing
# )

# #%% ============================================================
# # EARLY STOPPING SETUP
# # ============================================================
# best_val_acc = 0.0
# best_val_loss = float('inf')
# best_model_state = None
# best_epoch = 0
# patience_counter = 0

# # Training history tracking
# train_history = {'loss': [], 'accuracy': []}
# val_history = {'loss': [], 'accuracy': []}

# #%% ============================================================
# # TRAINING LOOP
# # ============================================================
# print("\n" + "="*50)
# print("TRAINING")
# print("="*50)
# print(f"Max Epochs: {EPOCHS}")
# print(f"Early Stopping Patience: {PATIENCE}")
# print(f"Batch Size: {BATCH_SIZE}")
# print(f"Learning Rate: {LEARNING_RATE}")
# print("="*50)

# for epoch in range(EPOCHS):
#     train_sampler.set_epoch(epoch)
#     print(f"\nEpoch {epoch+1}/{EPOCHS}")
#     print("-" * 40)    

#     # ---- Training ----
#     model.train()
#     train_loss = 0.0
#     train_correct = 0
#     train_total = 0

#     for batch in tqdm(train_loader, desc="Training"):    
#         images = batch['topo'].to(device)
#         psds = batch['psd'].to(device)
#         acs = batch['ac'].to(device)
#         labels = batch['label'].to(device)
        
#         optimizer.zero_grad()
#         outputs = model(images, psds, acs)
#         labels_cls = torch.argmax(labels, dim=1)
#         loss = criterion(outputs, labels_cls)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item() * images.size(0)
#         _, predicted = torch.max(outputs, 1)
#         train_correct += (predicted == labels_cls).sum().item()
#         train_total += labels.size(0)

#     train_accuracy = train_correct / train_total
#     train_epoch_loss = train_loss / train_total
#     train_history['loss'].append(train_epoch_loss)
#     train_history['accuracy'].append(train_accuracy)
#     print(f"Train Loss: {train_epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}")

#     # ---- Validation ----
#     model.eval()
#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0

#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc="Validation"):
#             images = batch['topo'].to(device)
#             psds = batch['psd'].to(device)
#             acs = batch['ac'].to(device)
#             labels = batch['label'].to(device)
#             outputs = model(images, psds, acs)
#             labels_cls = torch.argmax(labels, dim=1)
#             loss = criterion(outputs, labels_cls)

#             val_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs, 1)
#             val_correct += (predicted == labels_cls).sum().item()
#             val_total += labels.size(0)

#     val_accuracy = val_correct / val_total
#     val_epoch_loss = val_loss / val_total
#     val_history['loss'].append(val_epoch_loss)
#     val_history['accuracy'].append(val_accuracy)
#     print(f"Val Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
#     # ---- Learning Rate Scheduler ----
#     scheduler.step(val_accuracy)
#     current_lr = optimizer.param_groups[0]['lr']
#     print(f"Learning Rate: {current_lr:.6f}")

#     # ---- Early Stopping Check ----
#     if val_accuracy > best_val_acc + MIN_DELTA:
#         best_val_acc = val_accuracy
#         best_val_loss = val_epoch_loss
#         best_epoch = epoch
#         best_model_state = copy.deepcopy(model.module.state_dict())
#         patience_counter = 0
#         print(f"★ New best validation accuracy: {val_accuracy:.4f}")
#     else:
#         patience_counter += 1
#         print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
        
#         if patience_counter >= PATIENCE:
#             print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
#             print(f"Best epoch was {best_epoch+1} with val accuracy {best_val_acc:.4f}")
#             break

# print("\n" + "="*50)
# print("TRAINING COMPLETE")
# print("="*50)
# print(f"Best Epoch: {best_epoch+1}")
# print(f"Best Validation Accuracy: {best_val_acc:.4f}")
# print(f"Best Validation Loss: {best_val_loss:.4f}")

# # Load best model
# if best_model_state is not None:
#     model.module.load_state_dict(best_model_state)
#     print("Loaded best model weights.")

# #%% ============================================================
# # EVALUATION AND EXPORT (Only on rank 0)
# # ============================================================
# if rank == 0:
#     print("\n" + "="*50)
#     print("FINAL MODEL EVALUATION (Best Model)")
#     print("="*50)
    
#     # Create output directory
#     output_dir = "../outputs"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Evaluate on validation set
#     y_true, y_pred, y_probs, metrics = evaluate_model(model, val_loader, device, class_names)
    
#     # Generate confusion matrix
#     cm = plot_confusion_matrix(y_true, y_pred, class_names, 
#                                save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
#     # Generate ROC curves
#     auc_scores = plot_roc_curves(y_true, y_probs, class_names,
#                                  save_path=os.path.join(output_dir, 'roc_curves.png'))
    
#     # Save training history (with best epoch marker)
#     save_model_diagnostics(train_history, val_history, best_epoch=best_epoch,
#                           save_path=os.path.join(output_dir, 'training_history.png'))
    
#     # Save metrics to CSV
#     actual_epochs = len(train_history['loss'])
#     metrics_df = pd.DataFrame({
#         'Epoch': list(range(1, actual_epochs + 1)),
#         'Train_Loss': train_history['loss'],
#         'Train_Accuracy': train_history['accuracy'],
#         'Val_Loss': val_history['loss'],
#         'Val_Accuracy': val_history['accuracy']
#     })
#     metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)
#     print(f"Training metrics saved to {os.path.join(output_dir, 'training_metrics.csv')}")
    
#     # Save classification metrics
#     class_metrics_data = []
#     for i, class_name in enumerate(class_names):
#         class_metrics_data.append({
#             'Class': class_name,
#             'Precision': metrics[i]['precision'],
#             'Recall': metrics[i]['recall'],
#             'F1-Score': metrics[i]['f1'],
#             'Support': int(metrics[i]['support'])
#         })
#     class_metrics_df = pd.DataFrame(class_metrics_data)
#     class_metrics_df.to_csv(os.path.join(output_dir, 'classification_metrics.csv'), index=False)
#     print(f"Classification metrics saved to {os.path.join(output_dir, 'classification_metrics.csv')}")
    
#     # Save AUC scores
#     auc_df = pd.DataFrame({
#         'Class': class_names + ['Micro-average'],
#         'AUC': [auc_scores[i] for i in range(len(class_names))] + [auc_scores['micro']]
#     })
#     auc_df.to_csv(os.path.join(output_dir, 'auc_scores.csv'), index=False)
#     print(f"AUC scores saved to {os.path.join(output_dir, 'auc_scores.csv')}")
    
#     # Save training summary
#     summary_df = pd.DataFrame({
#         'Metric': ['Best Epoch', 'Best Val Accuracy', 'Best Val Loss', 
#                    'Total Epochs Run', 'Random Seed', 'Early Stopping Patience'],
#         'Value': [best_epoch + 1, best_val_acc, best_val_loss, 
#                   actual_epochs, RANDOM_SEED, PATIENCE]
#     })
#     summary_df.to_csv(os.path.join(output_dir, 'training_summary.csv'), index=False)
#     print(f"Training summary saved to {os.path.join(output_dir, 'training_summary.csv')}")
    
#     # Export to ONNX
#     sample_batch = next(iter(val_loader))
#     input_shapes = {
#         'topo': sample_batch['topo'].shape[1:],
#         'psd': sample_batch['psd'].shape[1:],
#         'ac': sample_batch['ac'].shape[1:]
#     }
    
#     export_to_onnx(model, 
#                    os.path.join(output_dir, 'icmobi_model.onnx'),
#                    device,
#                    input_shapes)
    
#     # Save PyTorch model weights
#     torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
#                os.path.join(output_dir, 'icmobi_model_weights.pth'))
#     print(f"PyTorch weights saved to {os.path.join(output_dir, 'icmobi_model_weights.pth')}")
    
#     print("\n" + "="*50)
#     print("All outputs saved to:", output_dir)
#     print("="*50)
#     print("\nIMPROVEMENTS APPLIED:")
#     print("  ✓ Combined both datasets")
#     print("  ✓ Stratified split (class balance preserved)")
#     print("  ✓ Fixed random seed (reproducible)")
#     print("  ✓ Class weights (handles imbalance)")
#     print("  ✓ Early stopping (prevents overfitting)")
#     print("  ✓ LR scheduler (adaptive learning rate)")
#     print("  ✓ FIXED: Now works with LABEL_1-7 format!")
#     print("="*50)

# cleanup()




# # #2 class
# # import torch
# # import torch.distributed as dist
# # import torch.nn as nn
# # from torch.nn.parallel import DistributedDataParallel as DDP
# # from torch.utils.data import DataLoader, Subset
# # from torch.utils.data.distributed import DistributedSampler
# # from torchvision import transforms
# # from torch.optim.lr_scheduler import ReduceLROnPlateau
# # from tqdm import tqdm
# # from sklearn.model_selection import train_test_split
# # import copy
# # #--
# # from icmobi_ext import icmobi_model
# # from icmobi_ext import icmobi_dataloader
# # import pandas as pd
# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import torch.onnx

# # #%% SETTING UP MULTIP
# # print(f"[Rank {os.environ.get('RANK', '?')}] "
# #       f"MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}, "
# #       f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

# # #%% CPU INITIALIZATION
# # def setup_distributed(backend="gloo"):
# #     """Initialize torch.distributed under SLURM or torchrun."""
# #     import os
# #     import socket
# #     import torch.distributed as dist

# #     # ---- detect mode ----
# #     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
# #         rank = int(os.environ["RANK"])
# #         world_size = int(os.environ["WORLD_SIZE"])
# #         print(f"Detected torchrun launch: rank={rank}, world_size={world_size}")
# #     elif "SLURM_PROCID" in os.environ:
# #         rank = int(os.environ["SLURM_PROCID"])
# #         world_size = int(os.environ["SLURM_NTASKS"])
# #         os.environ["RANK"] = str(rank)
# #         os.environ["WORLD_SIZE"] = str(world_size)
# #         os.environ.setdefault("MASTER_ADDR", socket.gethostname())
# #         os.environ.setdefault("MASTER_PORT", "12355")
# #         print(f"Detected SLURM launch: rank={rank}, world_size={world_size}")
# #     else:
# #         raise EnvironmentError("Distributed setup failed: no RANK or SLURM_PROCID found")

# #     # ---- init process group ----
# #     dist.init_process_group(
# #         backend=backend,
# #         init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
# #         world_size=world_size,
# #         rank=rank,
# #     )
# #     print(f"[Rank {rank}] Initialized process group (world size={world_size})")
# #     return rank, world_size

# # def cleanup():
# #     dist.destroy_process_group()

# # #-- setup distributed computing for openmpi
# # rank, world_size = setup_distributed("gloo")
# # device = torch.device("cpu")

# # #%% MULTIPROCESSING/THREADING
# # print(torch.__config__.parallel_info())

# # #%% DATASET TRANSFORMS
# # transform_pipeline = transforms.Compose([
# #     icmobi_dataloader.RandomHorizontalFlipTopography(),
# #     icmobi_dataloader.RandomGaussianNoiseTopography(std=0.02),
# #     icmobi_dataloader.RandomTopographyDropout(max_size=6),
# #     icmobi_dataloader.NormalizeTopography(),
    
# #     icmobi_dataloader.AddNoisePSD(std=0.01),
# #     icmobi_dataloader.NormalizePSD(),

# #     icmobi_dataloader.AddNoiseAC(std=0.01),
# #     icmobi_dataloader.NormalizeAC()
# # ])
# # transform_pipeline = None

# # #%% EVALUATION FUNCTIONS (PyTorch Native)
# # def compute_confusion_matrix(y_true, y_pred, num_classes):
# #     """Compute confusion matrix using PyTorch"""
# #     cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
# #     for t, p in zip(y_true, y_pred):
# #         cm[t, p] += 1
# #     return cm

# # def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png', title='Confusion Matrix'):
# #     """Plot and save confusion matrix using PyTorch"""
# #     num_classes = len(class_names)
# #     cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    
# #     plt.figure(figsize=(10, 8))
# #     sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues', 
# #                 xticklabels=class_names, yticklabels=class_names)
# #     plt.title(title)
# #     plt.ylabel('True Label')
# #     plt.xlabel('Predicted Label')
# #     plt.tight_layout()
# #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     plt.close()
# #     print(f"Confusion matrix saved to {save_path}")
# #     return cm

# # def compute_roc_curve(y_true, y_scores, pos_label):
# #     """Compute ROC curve using PyTorch"""
# #     # Sort by score
# #     sorted_indices = torch.argsort(y_scores, descending=True)
# #     y_true_sorted = y_true[sorted_indices]
# #     y_scores_sorted = y_scores[sorted_indices]
    
# #     # Compute TPR and FPR at each threshold
# #     tps = torch.cumsum(y_true_sorted, dim=0)
# #     fps = torch.cumsum(1 - y_true_sorted, dim=0)
    
# #     total_pos = tps[-1]
# #     total_neg = fps[-1]
    
# #     tpr = tps.float() / total_pos.float() if total_pos > 0 else torch.zeros_like(tps).float()
# #     fpr = fps.float() / total_neg.float() if total_neg > 0 else torch.zeros_like(fps).float()
    
# #     # Add (0,0) and (1,1) points
# #     fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])
# #     tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
    
# #     return fpr, tpr

# # def compute_auc(fpr, tpr):
# #     """Compute AUC using trapezoidal rule"""
# #     # Sort by fpr
# #     sorted_indices = torch.argsort(fpr)
# #     fpr_sorted = fpr[sorted_indices]
# #     tpr_sorted = tpr[sorted_indices]
    
# #     # Trapezoidal integration
# #     auc = torch.trapz(tpr_sorted, fpr_sorted)
# #     return auc.item()

# # def plot_roc_curves(y_true, y_probs, class_names, save_path='roc_curves.png', title='ROC Curves'):
# #     """Plot ROC curves for multi-class classification using PyTorch"""
# #     num_classes = len(class_names)
    
# #     # Convert to one-hot
# #     y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=num_classes)
    
# #     fpr_dict = {}
# #     tpr_dict = {}
# #     auc_dict = {}
    
# #     # Compute ROC for each class
# #     for i in range(num_classes):
# #         fpr, tpr = compute_roc_curve(y_true_onehot[:, i], y_probs[:, i], pos_label=1)
# #         fpr_dict[i] = fpr
# #         tpr_dict[i] = tpr
# #         auc_dict[i] = compute_auc(fpr, tpr)
    
# #     # Compute micro-average
# #     y_true_flat = y_true_onehot.reshape(-1)
# #     y_probs_flat = y_probs.reshape(-1)
# #     fpr_micro, tpr_micro = compute_roc_curve(y_true_flat, y_probs_flat, pos_label=1)
# #     auc_dict['micro'] = compute_auc(fpr_micro, tpr_micro)
    
# #     # Plot
# #     plt.figure(figsize=(10, 8))
# #     colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
# #     for i, color in zip(range(num_classes), colors):
# #         plt.plot(fpr_dict[i].numpy(), tpr_dict[i].numpy(), color=color, lw=2,
# #                 label=f'{class_names[i]} (AUC = {auc_dict[i]:.3f})')
    
# #     plt.plot(fpr_micro.numpy(), tpr_micro.numpy(), color='deeppink', linestyle='--', lw=2,
# #             label=f'Micro-average (AUC = {auc_dict["micro"]:.3f})')
    
# #     plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
# #     plt.xlim([0.0, 1.0])
# #     plt.ylim([0.0, 1.05])
# #     plt.xlabel('False Positive Rate')
# #     plt.ylabel('True Positive Rate')
# #     plt.title(title)
# #     plt.legend(loc="lower right")
# #     plt.grid(alpha=0.3)
# #     plt.tight_layout()
# #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     plt.close()
# #     print(f"ROC curves saved to {save_path}")
    
# #     return auc_dict

# # def compute_classification_metrics(y_true, y_pred, num_classes):
# #     """Compute precision, recall, F1 for each class using PyTorch"""
# #     metrics = {}
    
# #     for i in range(num_classes):
# #         # Binary masks for class i
# #         true_pos = ((y_true == i) & (y_pred == i)).sum().float()
# #         false_pos = ((y_true != i) & (y_pred == i)).sum().float()
# #         false_neg = ((y_true == i) & (y_pred != i)).sum().float()
# #         true_neg = ((y_true != i) & (y_pred != i)).sum().float()
        
# #         # Precision, Recall, F1
# #         precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else torch.tensor(0.0)
# #         recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else torch.tensor(0.0)
# #         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
        
# #         support = (y_true == i).sum()
        
# #         metrics[i] = {
# #             'precision': precision.item(),
# #             'recall': recall.item(),
# #             'f1': f1.item(),
# #             'support': support.item()
# #         }
    
# #     # Compute macro and weighted averages
# #     precisions = [metrics[i]['precision'] for i in range(num_classes)]
# #     recalls = [metrics[i]['recall'] for i in range(num_classes)]
# #     f1s = [metrics[i]['f1'] for i in range(num_classes)]
# #     supports = [metrics[i]['support'] for i in range(num_classes)]
# #     total_support = sum(supports)
    
# #     metrics['macro_avg'] = {
# #         'precision': np.mean(precisions),
# #         'recall': np.mean(recalls),
# #         'f1': np.mean(f1s),
# #         'support': total_support
# #     }
    
# #     metrics['weighted_avg'] = {
# #         'precision': np.average(precisions, weights=supports) if total_support > 0 else 0,
# #         'recall': np.average(recalls, weights=supports) if total_support > 0 else 0,
# #         'f1': np.average(f1s, weights=supports) if total_support > 0 else 0,
# #         'support': total_support
# #     }
    
# #     # Compute balanced accuracy (average of recalls)
# #     metrics['balanced_accuracy'] = np.mean(recalls)
    
# #     return metrics

# # def compute_balanced_accuracy(y_true, y_pred, num_classes):
# #     """Compute balanced accuracy (average of per-class recalls)"""
# #     recalls = []
# #     for i in range(num_classes):
# #         true_pos = ((y_true == i) & (y_pred == i)).sum().float()
# #         false_neg = ((y_true == i) & (y_pred != i)).sum().float()
# #         recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else torch.tensor(0.0)
# #         recalls.append(recall.item())
# #     return np.mean(recalls)

# # def print_classification_report(metrics, class_names, title="Classification Report"):
# #     """Print classification report similar to sklearn"""
# #     print(f"\n{title}:")
# #     print("-" * 70)
# #     print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
# #     print("-" * 70)
    
# #     for i, class_name in enumerate(class_names):
# #         m = metrics[i]
# #         print(f"{class_name:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
    
# #     print("-" * 70)
# #     m = metrics['macro_avg']
# #     print(f"{'Macro avg':<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
    
# #     m = metrics['weighted_avg']
# #     print(f"{'Weighted avg':<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
    
# #     print("-" * 70)
# #     print(f"{'Balanced Accuracy':<20} {metrics['balanced_accuracy']:<12.3f}")
# #     print("-" * 70)

# # def save_model_diagnostics(train_history, val_history, best_epoch=None, save_path='training_history.png'):
# #     """Plot and save training diagnostics"""
# #     fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
# #     # Loss plot
# #     axes[0].plot(train_history['loss'], label='Train Loss', marker='o')
# #     axes[0].plot(val_history['loss'], label='Val Loss', marker='s')
# #     if best_epoch is not None:
# #         axes[0].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
# #     axes[0].set_xlabel('Epoch')
# #     axes[0].set_ylabel('Loss')
# #     axes[0].set_title('Training and Validation Loss')
# #     axes[0].legend()
# #     axes[0].grid(alpha=0.3)
    
# #     # Accuracy plot
# #     axes[1].plot(train_history['accuracy'], label='Train Accuracy', marker='o')
# #     axes[1].plot(val_history['accuracy'], label='Val Accuracy', marker='s')
# #     if best_epoch is not None:
# #         axes[1].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
# #     axes[1].set_xlabel('Epoch')
# #     axes[1].set_ylabel('Accuracy')
# #     axes[1].set_title('Training and Validation Accuracy')
# #     axes[1].legend()
# #     axes[1].grid(alpha=0.3)
    
# #     plt.tight_layout()
# #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     plt.close()
# #     print(f"Training history saved to {save_path}")

# # #%% ============================================================
# # # 2-CLASS EVALUATION FUNCTIONS (Brain vs Other)
# # # ============================================================
# # def convert_to_2class(y_7class):
# #     """
# #     Convert 7-class labels to 2-class (Brain vs Other)
# #     Brain = 0, Other = 1 (all non-Brain classes)
    
# #     Original classes:
# #     0: Brain -> 0 (Brain)
# #     1: Muscle -> 1 (Other)
# #     2: Eye -> 1 (Other)
# #     3: Heart -> 1 (Other)
# #     4: Line_Noise -> 1 (Other)
# #     5: Channel_Noise -> 1 (Other)
# #     6: Other -> 1 (Other)
# #     """
# #     y_2class = (y_7class != 0).long()  # 0 = Brain, 1 = Other
# #     return y_2class

# # def convert_probs_to_2class(probs_7class):
# #     """
# #     Convert 7-class probabilities to 2-class probabilities
# #     Brain prob = prob[0]
# #     Other prob = sum(prob[1:7])
# #     """
# #     brain_prob = probs_7class[:, 0:1]  # Keep dimension
# #     other_prob = probs_7class[:, 1:].sum(dim=1, keepdim=True)
# #     probs_2class = torch.cat([brain_prob, other_prob], dim=1)
# #     return probs_2class

# # def evaluate_2class(y_true_7class, y_pred_7class, y_probs_7class, output_dir, class_names_2class=['Brain', 'Other']):
# #     """
# #     Evaluate model performance in 2-class setting (Brain vs Other)
# #     This matches ICLabel's 2-class evaluation in the paper
# #     """
# #     print("\n" + "="*60)
# #     print("2-CLASS EVALUATION (Brain vs Other) - Matching ICLabel Paper")
# #     print("="*60)
    
# #     # Convert to 2-class
# #     y_true_2class = convert_to_2class(y_true_7class)
# #     y_pred_2class = convert_to_2class(y_pred_7class)
# #     y_probs_2class = convert_probs_to_2class(y_probs_7class)
    
# #     # Compute metrics
# #     metrics_2class = compute_classification_metrics(y_true_2class, y_pred_2class, num_classes=2)
    
# #     # Print report
# #     print_classification_report(metrics_2class, class_names_2class, title="2-Class Classification Report (Brain vs Other)")
    
# #     # Compute and print cross-entropy (as in ICLabel paper)
# #     # Cross entropy: -sum(t_i * log(p_i))
# #     y_true_onehot = torch.nn.functional.one_hot(y_true_2class, num_classes=2).float()
# #     epsilon = 1e-7  # Prevent log(0)
# #     cross_entropy = -torch.sum(y_true_onehot * torch.log(y_probs_2class + epsilon)) / len(y_true_2class)
# #     print(f"\n2-Class Cross Entropy: {cross_entropy.item():.4f}")
# #     print(f"(ICLabel achieved: 0.342)")
    
# #     # Generate confusion matrix
# #     cm_2class = plot_confusion_matrix(
# #         y_true_2class, y_pred_2class, class_names_2class,
# #         save_path=os.path.join(output_dir, 'confusion_matrix_2class.png'),
# #         title='2-Class Confusion Matrix (Brain vs Other)'
# #     )
    
# #     # Generate ROC curves
# #     auc_2class = plot_roc_curves(
# #         y_true_2class, y_probs_2class, class_names_2class,
# #         save_path=os.path.join(output_dir, 'roc_curves_2class.png'),
# #         title='2-Class ROC Curves (Brain vs Other)'
# #     )
    
# #     # Print comparison with ICLabel
# #     print("\n" + "-"*60)
# #     print("COMPARISON WITH ICLABEL (from paper Table 1):")
# #     print("-"*60)
# #     print(f"{'Metric':<30} {'ICMoBI (Ours)':<15} {'ICLabel':<15}")
# #     print("-"*60)
# #     print(f"{'2-Class Balanced Accuracy':<30} {metrics_2class['balanced_accuracy']:.3f}{'':>10} 0.841")
# #     print(f"{'2-Class Cross Entropy':<30} {cross_entropy.item():.3f}{'':>10} 0.342")
# #     print(f"{'Brain AUC':<30} {auc_2class[0]:.3f}{'':>10} ~0.90")
# #     print(f"{'Other AUC':<30} {auc_2class[1]:.3f}{'':>10} ~0.90")
# #     print("-"*60)
    
# #     return metrics_2class, auc_2class, cross_entropy.item()

# # def evaluate_model(model, data_loader, device, class_names):
# #     """Comprehensive model evaluation using PyTorch"""
# #     model.eval()
# #     all_preds = []
# #     all_labels = []
# #     all_probs = []
    
# #     with torch.no_grad():
# #         for batch in tqdm(data_loader, desc="Evaluating"):
# #             images = batch['topo'].to(device)
# #             psds = batch['psd'].to(device)
# #             acs = batch['ac'].to(device)
# #             labels = batch['label'].to(device)
            
# #             outputs = model(images, psds, acs)
# #             probs = torch.softmax(outputs, dim=1)
# #             _, predicted = torch.max(outputs, 1)
# #             labels_cls = torch.argmax(labels, dim=1)
            
# #             all_preds.append(predicted.cpu())
# #             all_labels.append(labels_cls.cpu())
# #             all_probs.append(probs.cpu())
    
# #     all_preds = torch.cat(all_preds)
# #     all_labels = torch.cat(all_labels)
# #     all_probs = torch.cat(all_probs)
    
# #     # Compute and print classification metrics
# #     num_classes = len(class_names)
# #     metrics = compute_classification_metrics(all_labels, all_preds, num_classes)
# #     print_classification_report(metrics, class_names, title="7-Class Classification Report")
    
# #     return all_labels, all_preds, all_probs, metrics

# # def export_to_onnx(model, save_path, device, input_shapes):
# #     """Export model to ONNX format"""
# #     model.eval()
    
# #     # Create dummy inputs
# #     dummy_topo = torch.randn(1, *input_shapes['topo']).to(device)
# #     dummy_psd = torch.randn(1, *input_shapes['psd']).to(device)
# #     dummy_ac = torch.randn(1, *input_shapes['ac']).to(device)
    
# #     # Export
# #     torch.onnx.export(
# #         model.module if isinstance(model, DDP) else model,
# #         (dummy_topo, dummy_psd, dummy_ac),
# #         save_path,
# #         export_params=True,
# #         opset_version=11,
# #         do_constant_folding=True,
# #         input_names=['topography', 'psd', 'autocorrelation'],
# #         output_names=['output'],
# #         dynamic_axes={
# #             'topography': {0: 'batch_size'},
# #             'psd': {0: 'batch_size'},
# #             'autocorrelation': {0: 'batch_size'},
# #             'output': {0: 'batch_size'}
# #         }
# #     )
# #     print(f"Model exported to ONNX: {save_path}")

# # #%% ============================================================
# # # CONFIGURATION
# # # ============================================================
# # # Hyperparameters
# # BATCH_SIZE = 32
# # EPOCHS = 20                # Reduced from 30 (early stopping will handle it)
# # LEARNING_RATE = 1e-4
# # VALID_SPLIT = 0.2
# # RANDOM_SEED = 42           # Fixed seed for reproducibility

# # # Early Stopping Configuration
# # PATIENCE = 10              # Stop if no improvement for 10 epochs
# # MIN_DELTA = 0.001          # Minimum improvement to count

# # # Class names (order matches LABEL_1 through LABEL_7)
# # class_names_7class = ['Brain', 'Muscle', 'Eye', 'Heart', 'Line_Noise', 'Channel_Noise', 'Other']
# # class_names_2class = ['Brain', 'Other']

# # # Label columns in the CSV
# # LABEL_COLS = ['LABEL_1', 'LABEL_2', 'LABEL_3', 'LABEL_4', 'LABEL_5', 'LABEL_6', 'LABEL_7']

# # #%% ============================================================
# # # LOAD AND COMBINE DATASETS
# # # ============================================================
# # print("\n" + "="*50)
# # print("LOADING DATASETS")
# # print("="*50)

# # # Device configuration
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Device: {device}")

# # # Load both datasets
# # rel_path = os.path.abspath("../data")

# # df1 = pd.read_csv(os.path.join(rel_path, "trainset_labeled.csv"))
# # print(f"Dataset 1: {len(df1)} samples")

# # df2 = pd.read_csv(os.path.join(rel_path, "trainset_labeled_2.csv"))
# # print(f"Dataset 2: {len(df2)} samples")

# # # Combine datasets
# # df = pd.concat([df1, df2], ignore_index=True)
# # print(f"Combined dataset: {len(df)} samples")

# # #%% ============================================================
# # # CREATE ic_label FROM LABEL_1-7 (FIXED!)
# # # ============================================================
# # # Convert soft labels (LABEL_1-7) to hard labels (ic_label)
# # # ic_label = name of the class with highest probability

# # print("\nConverting soft labels to hard labels...")
# # label_matrix = df[LABEL_COLS].values  # Shape: (N, 7)
# # hard_label_indices = np.argmax(label_matrix, axis=1)  # Get index of max probability
# # df['ic_label'] = [class_names_7class[i] for i in hard_label_indices]  # Map to class name

# # # Print 7-class distribution
# # print("\n7-Class Distribution:")
# # print("-" * 40)
# # class_counts = df['ic_label'].value_counts()
# # for class_name in class_names_7class:
# #     count = class_counts.get(class_name, 0)
# #     print(f"  {class_name:<15}: {count:5} samples ({100*count/len(df):5.1f}%)")
# # print("-" * 40)

# # # Print 2-class distribution
# # brain_count = class_counts.get('Brain', 0)
# # other_count = len(df) - brain_count
# # print("\n2-Class Distribution (Brain vs Other):")
# # print("-" * 40)
# # print(f"  {'Brain':<15}: {brain_count:5} samples ({100*brain_count/len(df):5.1f}%)")
# # print(f"  {'Other':<15}: {other_count:5} samples ({100*other_count/len(df):5.1f}%)")
# # print("-" * 40)

# # #%% ============================================================
# # # STRATIFIED SPLIT WITH FIXED SEED
# # # ============================================================
# # print("\n" + "="*50)
# # print("CREATING STRATIFIED TRAIN/VAL SPLIT")
# # print("="*50)

# # # Get labels for stratification
# # labels = df['ic_label'].values

# # # Create indices
# # indices = np.arange(len(df))

# # # Stratified split with fixed seed
# # train_idx, val_idx = train_test_split(
# #     indices,
# #     test_size=VALID_SPLIT,
# #     random_state=RANDOM_SEED,    # Fixed seed for reproducibility
# #     stratify=labels               # Maintains class proportions
# # )

# # print(f"Training samples: {len(train_idx)} ({100*len(train_idx)/len(df):.1f}%)")
# # print(f"Validation samples: {len(val_idx)} ({100*len(val_idx)/len(df):.1f}%)")

# # # Verify stratification worked
# # print("\nVerifying stratification (should be ~same proportions):")
# # train_labels = labels[train_idx]
# # val_labels = labels[val_idx]
# # for class_name in class_names_7class:
# #     train_pct = 100 * np.sum(train_labels == class_name) / len(train_labels)
# #     val_pct = 100 * np.sum(val_labels == class_name) / len(val_labels)
# #     print(f"  {class_name:<15}: Train {train_pct:5.1f}% | Val {val_pct:5.1f}%")

# # #%% ============================================================
# # # CREATE DATASETS AND DATALOADERS
# # # ============================================================
# # # Create full dataset
# # dataset = icmobi_dataloader.ICMOBIDatasetFormatter(df, transform=transform_pipeline)

# # # Create subset datasets using stratified indices
# # train_dataset = Subset(dataset, train_idx)
# # val_dataset = Subset(dataset, val_idx)

# # # Use distributed dataloading
# # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
# # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

# # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
# # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# # #%% ============================================================
# # # COMPUTE CLASS WEIGHTS FOR IMBALANCED DATA
# # # ============================================================
# # print("\n" + "="*50)
# # print("COMPUTING CLASS WEIGHTS")
# # print("="*50)

# # # Count samples per class in training set
# # train_class_counts = []
# # for class_name in class_names_7class:
# #     count = np.sum(train_labels == class_name)
# #     train_class_counts.append(max(count, 1))  # Avoid division by zero
    
# # train_class_counts = np.array(train_class_counts)

# # # Compute weights (inverse frequency, normalized)
# # class_weights = 1.0 / train_class_counts
# # class_weights = class_weights / class_weights.sum() * len(class_names_7class)
# # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# # print("Class weights (higher = more emphasis on this class):")
# # for i, class_name in enumerate(class_names_7class):
# #     print(f"  {class_name:<15}: {class_weights[i].item():.3f} (samples: {train_class_counts[i]})")

# # #%% ============================================================
# # # LOAD MODEL
# # # ============================================================
# # print("\n" + "="*50)
# # print("LOADING MODEL")
# # print("="*50)

# # rel_path = os.path.abspath("../icmobi_ext/utils/")
# # model = icmobi_model.ICMoBiNetTransferL(mat_path=os.path.join(rel_path, "netICL.mat"))

# # # Model to device    
# # model.to(device)
# # model = DDP(model)

# # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# # print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# # #%% ============================================================
# # # LOSS, OPTIMIZER, SCHEDULER
# # # ============================================================
# # # Loss with class weights (handles imbalanced data)
# # criterion = nn.CrossEntropyLoss(weight=class_weights)

# # # Optimizer
# # optimizer = torch.optim.Adam(
# #     filter(lambda p: p.requires_grad, model.parameters()), 
# #     lr=LEARNING_RATE
# # )

# # # Learning rate scheduler (reduces LR when validation plateaus)
# # scheduler = ReduceLROnPlateau(
# #     optimizer, 
# #     mode='max',           # Maximize validation accuracy
# #     factor=0.5,           # Reduce LR by half
# #     patience=5           # Wait 5 epochs before reducing
# # )

# # #%% ============================================================
# # # EARLY STOPPING SETUP
# # # ============================================================
# # best_val_acc = 0.0
# # best_val_loss = float('inf')
# # best_model_state = None
# # best_epoch = 0
# # patience_counter = 0

# # # Training history tracking
# # train_history = {'loss': [], 'accuracy': []}
# # val_history = {'loss': [], 'accuracy': []}

# # #%% ============================================================
# # # TRAINING LOOP
# # # ============================================================
# # print("\n" + "="*50)
# # print("TRAINING")
# # print("="*50)
# # print(f"Max Epochs: {EPOCHS}")
# # print(f"Early Stopping Patience: {PATIENCE}")
# # print(f"Batch Size: {BATCH_SIZE}")
# # print(f"Learning Rate: {LEARNING_RATE}")
# # print("="*50)

# # for epoch in range(EPOCHS):
# #     train_sampler.set_epoch(epoch)
# #     print(f"\nEpoch {epoch+1}/{EPOCHS}")
# #     print("-" * 40)    

# #     # ---- Training ----
# #     model.train()
# #     train_loss = 0.0
# #     train_correct = 0
# #     train_total = 0

# #     for batch in tqdm(train_loader, desc="Training"):    
# #         images = batch['topo'].to(device)
# #         psds = batch['psd'].to(device)
# #         acs = batch['ac'].to(device)
# #         labels = batch['label'].to(device)
        
# #         optimizer.zero_grad()
# #         outputs = model(images, psds, acs)
# #         labels_cls = torch.argmax(labels, dim=1)
# #         loss = criterion(outputs, labels_cls)
# #         loss.backward()
# #         optimizer.step()

# #         train_loss += loss.item() * images.size(0)
# #         _, predicted = torch.max(outputs, 1)
# #         train_correct += (predicted == labels_cls).sum().item()
# #         train_total += labels.size(0)

# #     train_accuracy = train_correct / train_total
# #     train_epoch_loss = train_loss / train_total
# #     train_history['loss'].append(train_epoch_loss)
# #     train_history['accuracy'].append(train_accuracy)
# #     print(f"Train Loss: {train_epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}")

# #     # ---- Validation ----
# #     model.eval()
# #     val_loss = 0.0
# #     val_correct = 0
# #     val_total = 0

# #     with torch.no_grad():
# #         for batch in tqdm(val_loader, desc="Validation"):
# #             images = batch['topo'].to(device)
# #             psds = batch['psd'].to(device)
# #             acs = batch['ac'].to(device)
# #             labels = batch['label'].to(device)
# #             outputs = model(images, psds, acs)
# #             labels_cls = torch.argmax(labels, dim=1)
# #             loss = criterion(outputs, labels_cls)

# #             val_loss += loss.item() * images.size(0)
# #             _, predicted = torch.max(outputs, 1)
# #             val_correct += (predicted == labels_cls).sum().item()
# #             val_total += labels.size(0)

# #     val_accuracy = val_correct / val_total
# #     val_epoch_loss = val_loss / val_total
# #     val_history['loss'].append(val_epoch_loss)
# #     val_history['accuracy'].append(val_accuracy)
# #     print(f"Val Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
# #     # ---- Learning Rate Scheduler ----
# #     scheduler.step(val_accuracy)
# #     current_lr = optimizer.param_groups[0]['lr']
# #     print(f"Learning Rate: {current_lr:.6f}")

# #     # ---- Early Stopping Check ----
# #     if val_accuracy > best_val_acc + MIN_DELTA:
# #         best_val_acc = val_accuracy
# #         best_val_loss = val_epoch_loss
# #         best_epoch = epoch
# #         best_model_state = copy.deepcopy(model.module.state_dict())
# #         patience_counter = 0
# #         print(f"★ New best validation accuracy: {val_accuracy:.4f}")
# #     else:
# #         patience_counter += 1
# #         print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
        
# #         if patience_counter >= PATIENCE:
# #             print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
# #             print(f"Best epoch was {best_epoch+1} with val accuracy {best_val_acc:.4f}")
# #             break

# # print("\n" + "="*50)
# # print("TRAINING COMPLETE")
# # print("="*50)
# # print(f"Best Epoch: {best_epoch+1}")
# # print(f"Best Validation Accuracy: {best_val_acc:.4f}")
# # print(f"Best Validation Loss: {best_val_loss:.4f}")

# # # Load best model
# # if best_model_state is not None:
# #     model.module.load_state_dict(best_model_state)
# #     print("Loaded best model weights.")

# # #%% ============================================================
# # # EVALUATION AND EXPORT (Only on rank 0)
# # # ============================================================
# # if rank == 0:
# #     print("\n" + "="*60)
# #     print("FINAL MODEL EVALUATION (Best Model)")
# #     print("="*60)
    
# #     # Create output directory
# #     output_dir = "../outputs"
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     # ============================================================
# #     # 7-CLASS EVALUATION
# #     # ============================================================
# #     print("\n" + "="*60)
# #     print("7-CLASS EVALUATION")
# #     print("="*60)
    
# #     # Evaluate on validation set
# #     y_true, y_pred, y_probs, metrics_7class = evaluate_model(model, val_loader, device, class_names_7class)
    
# #     # Compute 7-class cross entropy
# #     y_true_onehot_7 = torch.nn.functional.one_hot(y_true, num_classes=7).float()
# #     epsilon = 1e-7
# #     cross_entropy_7class = -torch.sum(y_true_onehot_7 * torch.log(y_probs + epsilon)) / len(y_true)
# #     print(f"\n7-Class Cross Entropy: {cross_entropy_7class.item():.4f}")
# #     print(f"(ICLabel achieved: 1.251)")
    
# #     # Generate 7-class confusion matrix
# #     cm_7class = plot_confusion_matrix(
# #         y_true, y_pred, class_names_7class, 
# #         save_path=os.path.join(output_dir, 'confusion_matrix_7class.png'),
# #         title='7-Class Confusion Matrix'
# #     )
    
# #     # Generate 7-class ROC curves
# #     auc_scores_7class = plot_roc_curves(
# #         y_true, y_probs, class_names_7class,
# #         save_path=os.path.join(output_dir, 'roc_curves_7class.png'),
# #         title='7-Class ROC Curves'
# #     )
    
# #     # ============================================================
# #     # 2-CLASS EVALUATION (Brain vs Other)
# #     # ============================================================
# #     metrics_2class, auc_2class, cross_entropy_2class = evaluate_2class(
# #         y_true, y_pred, y_probs, output_dir, class_names_2class
# #     )
    
# #     # ============================================================
# #     # SAVE ALL RESULTS
# #     # ============================================================
    
# #     # Save training history (with best epoch marker)
# #     save_model_diagnostics(train_history, val_history, best_epoch=best_epoch,
# #                           save_path=os.path.join(output_dir, 'training_history.png'))
    
# #     # Save training metrics to CSV
# #     actual_epochs = len(train_history['loss'])
# #     metrics_df = pd.DataFrame({
# #         'Epoch': list(range(1, actual_epochs + 1)),
# #         'Train_Loss': train_history['loss'],
# #         'Train_Accuracy': train_history['accuracy'],
# #         'Val_Loss': val_history['loss'],
# #         'Val_Accuracy': val_history['accuracy']
# #     })
# #     metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)
# #     print(f"Training metrics saved to {os.path.join(output_dir, 'training_metrics.csv')}")
    
# #     # Save 7-class classification metrics
# #     class_metrics_data_7 = []
# #     for i, class_name in enumerate(class_names_7class):
# #         class_metrics_data_7.append({
# #             'Class': class_name,
# #             'Precision': metrics_7class[i]['precision'],
# #             'Recall': metrics_7class[i]['recall'],
# #             'F1-Score': metrics_7class[i]['f1'],
# #             'Support': int(metrics_7class[i]['support'])
# #         })
# #     class_metrics_df_7 = pd.DataFrame(class_metrics_data_7)
# #     class_metrics_df_7.to_csv(os.path.join(output_dir, 'classification_metrics_7class.csv'), index=False)
# #     print(f"7-class metrics saved to {os.path.join(output_dir, 'classification_metrics_7class.csv')}")
    
# #     # Save 2-class classification metrics
# #     class_metrics_data_2 = []
# #     for i, class_name in enumerate(class_names_2class):
# #         class_metrics_data_2.append({
# #             'Class': class_name,
# #             'Precision': metrics_2class[i]['precision'],
# #             'Recall': metrics_2class[i]['recall'],
# #             'F1-Score': metrics_2class[i]['f1'],
# #             'Support': int(metrics_2class[i]['support'])
# #         })
# #     class_metrics_df_2 = pd.DataFrame(class_metrics_data_2)
# #     class_metrics_df_2.to_csv(os.path.join(output_dir, 'classification_metrics_2class.csv'), index=False)
# #     print(f"2-class metrics saved to {os.path.join(output_dir, 'classification_metrics_2class.csv')}")
    
# #     # Save 7-class AUC scores
# #     auc_df_7 = pd.DataFrame({
# #         'Class': class_names_7class + ['Micro-average'],
# #         'AUC': [auc_scores_7class[i] for i in range(len(class_names_7class))] + [auc_scores_7class['micro']]
# #     })
# #     auc_df_7.to_csv(os.path.join(output_dir, 'auc_scores_7class.csv'), index=False)
# #     print(f"7-class AUC scores saved to {os.path.join(output_dir, 'auc_scores_7class.csv')}")
    
# #     # Save 2-class AUC scores
# #     auc_df_2 = pd.DataFrame({
# #         'Class': class_names_2class + ['Micro-average'],
# #         'AUC': [auc_2class[i] for i in range(len(class_names_2class))] + [auc_2class['micro']]
# #     })
# #     auc_df_2.to_csv(os.path.join(output_dir, 'auc_scores_2class.csv'), index=False)
# #     print(f"2-class AUC scores saved to {os.path.join(output_dir, 'auc_scores_2class.csv')}")
    
# #     # Save comprehensive training summary
# #     summary_df = pd.DataFrame({
# #         'Metric': [
# #             'Best Epoch', 
# #             'Best Val Accuracy (7-class)', 
# #             'Best Val Loss',
# #             'Total Epochs Run', 
# #             'Random Seed', 
# #             'Early Stopping Patience',
# #             '7-Class Balanced Accuracy',
# #             '7-Class Cross Entropy',
# #             '7-Class Weighted F1',
# #             '2-Class Balanced Accuracy',
# #             '2-Class Cross Entropy',
# #             '2-Class Brain F1',
# #             '2-Class Other F1',
# #             'ICLabel 7-Class Balanced Acc (Reference)',
# #             'ICLabel 2-Class Balanced Acc (Reference)',
# #             'ICLabel 7-Class Cross Entropy (Reference)',
# #             'ICLabel 2-Class Cross Entropy (Reference)'
# #         ],
# #         'Value': [
# #             best_epoch + 1, 
# #             best_val_acc, 
# #             best_val_loss,
# #             actual_epochs, 
# #             RANDOM_SEED, 
# #             PATIENCE,
# #             metrics_7class['balanced_accuracy'],
# #             cross_entropy_7class.item(),
# #             metrics_7class['weighted_avg']['f1'],
# #             metrics_2class['balanced_accuracy'],
# #             cross_entropy_2class,
# #             metrics_2class[0]['f1'],
# #             metrics_2class[1]['f1'],
# #             0.597,  # ICLabel reference
# #             0.841,  # ICLabel reference
# #             1.251,  # ICLabel reference
# #             0.342   # ICLabel reference
# #         ]
# #     })
# #     summary_df.to_csv(os.path.join(output_dir, 'training_summary.csv'), index=False)
# #     print(f"Training summary saved to {os.path.join(output_dir, 'training_summary.csv')}")
    
# #     # Export to ONNX
# #     sample_batch = next(iter(val_loader))
# #     input_shapes = {
# #         'topo': sample_batch['topo'].shape[1:],
# #         'psd': sample_batch['psd'].shape[1:],
# #         'ac': sample_batch['ac'].shape[1:]
# #     }
    
# #     export_to_onnx(model, 
# #                    os.path.join(output_dir, 'icmobi_model.onnx'),
# #                    device,
# #                    input_shapes)
    
# #     # Save PyTorch model weights
# #     torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
# #                os.path.join(output_dir, 'icmobi_model_weights.pth'))
# #     print(f"PyTorch weights saved to {os.path.join(output_dir, 'icmobi_model_weights.pth')}")
    
# #     # ============================================================
# #     # FINAL SUMMARY COMPARISON WITH ICLABEL
# #     # ============================================================
# #     print("\n" + "="*70)
# #     print("FINAL SUMMARY: ICMoBI vs ICLabel")
# #     print("="*70)
# #     print(f"{'Metric':<40} {'ICMoBI (Ours)':<15} {'ICLabel':<15}")
# #     print("-"*70)
# #     print(f"{'7-Class Balanced Accuracy':<40} {metrics_7class['balanced_accuracy']:.3f}{'':>10} 0.597")
# #     print(f"{'7-Class Cross Entropy':<40} {cross_entropy_7class.item():.3f}{'':>10} 1.251")
# #     print(f"{'7-Class Weighted F1':<40} {metrics_7class['weighted_avg']['f1']:.3f}{'':>10} ~0.60")
# #     print("-"*70)
# #     print(f"{'2-Class Balanced Accuracy':<40} {metrics_2class['balanced_accuracy']:.3f}{'':>10} 0.841")
# #     print(f"{'2-Class Cross Entropy':<40} {cross_entropy_2class:.3f}{'':>10} 0.342")
# #     print(f"{'2-Class Brain Recall':<40} {metrics_2class[0]['recall']:.3f}{'':>10} ~0.84")
# #     print(f"{'2-Class Other Recall':<40} {metrics_2class[1]['recall']:.3f}{'':>10} ~0.84")
# #     print("="*70)
    
# #     print("\n" + "="*50)
# #     print("All outputs saved to:", output_dir)
# #     print("="*50)
# #     print("\nFILES GENERATED:")
# #     print("  ✓ confusion_matrix_7class.png")
# #     print("  ✓ confusion_matrix_2class.png")
# #     print("  ✓ roc_curves_7class.png")
# #     print("  ✓ roc_curves_2class.png")
# #     print("  ✓ training_history.png")
# #     print("  ✓ classification_metrics_7class.csv")
# #     print("  ✓ classification_metrics_2class.csv")
# #     print("  ✓ auc_scores_7class.csv")
# #     print("  ✓ auc_scores_2class.csv")
# #     print("  ✓ training_summary.csv")
# #     print("  ✓ icmobi_model.onnx")
# #     print("  ✓ icmobi_model_weights.pth")
# #     print("="*50)

# # cleanup()








#code with expert validation
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy
#--
from icmobi_ext import icmobi_model
from icmobi_ext import icmobi_dataloader
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.onnx

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
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"Detected torchrun launch: rank={rank}, world_size={world_size}")
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

#%% EVALUATION FUNCTIONS (PyTorch Native)
def compute_confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix using PyTorch"""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix using PyTorch"""
    num_classes = len(class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    return cm

def compute_roc_curve(y_true, y_scores, pos_label):
    """Compute ROC curve using PyTorch"""
    # Sort by score
    sorted_indices = torch.argsort(y_scores, descending=True)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Compute TPR and FPR at each threshold
    tps = torch.cumsum(y_true_sorted, dim=0)
    fps = torch.cumsum(1 - y_true_sorted, dim=0)
    
    total_pos = tps[-1]
    total_neg = fps[-1]
    
    tpr = tps.float() / total_pos.float() if total_pos > 0 else torch.zeros_like(tps).float()
    fpr = fps.float() / total_neg.float() if total_pos > 0 else torch.zeros_like(fps).float()
    
    # Add (0,0) and (1,1) points
    fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])
    tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
    
    return fpr, tpr

def compute_auc(fpr, tpr):
    """Compute AUC using trapezoidal rule"""
    # Sort by fpr
    sorted_indices = torch.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Trapezoidal integration
    auc = torch.trapz(tpr_sorted, fpr_sorted)
    return auc.item()

def plot_roc_curves(y_true, y_probs, class_names, save_path='roc_curves.png'):
    """Plot ROC curves for multi-class classification using PyTorch"""
    num_classes = len(class_names)
    
    # Convert to one-hot
    y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=num_classes)
    
    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}
    
    # Compute ROC for each class
    for i in range(num_classes):
        fpr, tpr = compute_roc_curve(y_true_onehot[:, i], y_probs[:, i], pos_label=1)
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        auc_dict[i] = compute_auc(fpr, tpr)
    
    # Compute micro-average
    y_true_flat = y_true_onehot.reshape(-1)
    y_probs_flat = y_probs.reshape(-1)
    fpr_micro, tpr_micro = compute_roc_curve(y_true_flat, y_probs_flat, pos_label=1)
    auc_dict['micro'] = compute_auc(fpr_micro, tpr_micro)
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr_dict[i].numpy(), tpr_dict[i].numpy(), color=color, lw=2,
                label=f'{class_names[i]} (AUC = {auc_dict[i]:.3f})')
    
    plt.plot(fpr_micro.numpy(), tpr_micro.numpy(), color='deeppink', linestyle='--', lw=2,
            label=f'Micro-average (AUC = {auc_dict["micro"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-Class Classification')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {save_path}")
    
    return auc_dict

def compute_classification_metrics(y_true, y_pred, num_classes):
    """Compute precision, recall, F1 for each class using PyTorch"""
    metrics = {}
    
    for i in range(num_classes):
        # Binary masks for class i
        true_pos = ((y_true == i) & (y_pred == i)).sum().float()
        false_pos = ((y_true != i) & (y_pred == i)).sum().float()
        false_neg = ((y_true == i) & (y_pred != i)).sum().float()
        true_neg = ((y_true != i) & (y_pred != i)).sum().float()
        
        # Precision, Recall, F1
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else torch.tensor(0.0)
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else torch.tensor(0.0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
        
        support = (y_true == i).sum()
        
        metrics[i] = {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'support': support.item()
        }
    
    # Compute macro and weighted averages
    precisions = [metrics[i]['precision'] for i in range(num_classes)]
    recalls = [metrics[i]['recall'] for i in range(num_classes)]
    f1s = [metrics[i]['f1'] for i in range(num_classes)]
    supports = [metrics[i]['support'] for i in range(num_classes)]
    total_support = sum(supports)
    
    metrics['macro_avg'] = {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s),
        'support': total_support
    }
    
    metrics['weighted_avg'] = {
        'precision': np.average(precisions, weights=supports) if total_support > 0 else 0,
        'recall': np.average(recalls, weights=supports) if total_support > 0 else 0,
        'f1': np.average(f1s, weights=supports) if total_support > 0 else 0,
        'support': total_support
    }
    
    return metrics

def print_classification_report(metrics, class_names):
    """Print classification report similar to sklearn"""
    print("\nClassification Report:")
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        m = metrics[i]
        print(f"{class_name:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
    
    print("-" * 70)
    m = metrics['macro_avg']
    print(f"{'Macro avg':<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
    
    m = metrics['weighted_avg']
    print(f"{'Weighted avg':<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {int(m['support']):<10}")
    print("-" * 70)

def save_model_diagnostics(train_history, val_history, best_epoch=None, save_path='training_history.png'):
    """Plot and save training diagnostics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(train_history['loss'], label='Train Loss', marker='o')
    axes[0].plot(val_history['loss'], label='Val Loss', marker='s')
    if best_epoch is not None:
        axes[0].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(train_history['accuracy'], label='Train Accuracy', marker='o')
    axes[1].plot(val_history['accuracy'], label='Val Accuracy (Expert)', marker='s')
    if best_epoch is not None:
        axes[1].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")

def evaluate_model(model, data_loader, device, class_names):
    """Comprehensive model evaluation using PyTorch"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch['topo'].to(device)
            psds = batch['psd'].to(device)
            acs = batch['ac'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, psds, acs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            labels_cls = torch.argmax(labels, dim=1)
            
            all_preds.append(predicted.cpu())
            all_labels.append(labels_cls.cpu())
            all_probs.append(probs.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # Compute and print classification metrics
    num_classes = len(class_names)
    metrics = compute_classification_metrics(all_labels, all_preds, num_classes)
    print_classification_report(metrics, class_names)
    
    return all_labels, all_preds, all_probs, metrics

def export_to_onnx(model, save_path, device, input_shapes):
    """Export model to ONNX format"""
    model.eval()
    
    # Create dummy inputs
    dummy_topo = torch.randn(1, *input_shapes['topo']).to(device)
    dummy_psd = torch.randn(1, *input_shapes['psd']).to(device)
    dummy_ac = torch.randn(1, *input_shapes['ac']).to(device)
    
    # Export
    torch.onnx.export(
        model.module if isinstance(model, DDP) else model,
        (dummy_topo, dummy_psd, dummy_ac),
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['topography', 'psd', 'autocorrelation'],
        output_names=['output'],
        dynamic_axes={
            'topography': {0: 'batch_size'},
            'psd': {0: 'batch_size'},
            'autocorrelation': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to ONNX: {save_path}")

#%% ============================================================
# CONFIGURATION
# ============================================================
# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20                # Reduced from 30 (early stopping will handle it)
LEARNING_RATE = 1e-4
RANDOM_SEED = 42           # Fixed seed for reproducibility

# Early Stopping Configuration
PATIENCE = 10              # Stop if no improvement for 10 epochs
MIN_DELTA = 0.001          # Minimum improvement to count

# Class names (order matches LABEL_1 through LABEL_7)
class_names = ['Brain', 'Muscle', 'Eye', 'Heart', 'Line_Noise', 'Channel_Noise', 'Other']

# Label columns in the CSV
LABEL_COLS = ['LABEL_1', 'LABEL_2', 'LABEL_3', 'LABEL_4', 'LABEL_5', 'LABEL_6', 'LABEL_7']

#%% ============================================================
# LOAD DATASETS
# ============================================================
print("\n" + "="*50)
print("LOADING DATASETS")
print("="*50)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load training datasets
rel_path = os.path.abspath("../data")

df1 = pd.read_csv(os.path.join(rel_path, "trainset_labeled.csv"))
print(f"Training Dataset 1: {len(df1)} samples")

df2 = pd.read_csv(os.path.join(rel_path, "trainset_labeled_2.csv"))
print(f"Training Dataset 2: {len(df2)} samples")

# Combine training datasets
train_df = pd.concat([df1, df2], ignore_index=True)
print(f"Combined training dataset: {len(train_df)} samples")

# Load expert-labeled validation dataset
val_df = pd.read_csv(os.path.join(rel_path, "trainset_labeled_expert.csv"))
print(f"Expert validation dataset: {len(val_df)} samples")

#%% ============================================================
# CREATE ic_label FROM LABEL_1-7
# ============================================================
print("\nConverting soft labels to hard labels...")

# Training set
train_label_matrix = train_df[LABEL_COLS].values
train_hard_indices = np.argmax(train_label_matrix, axis=1)
train_df['ic_label'] = [class_names[i] for i in train_hard_indices]

# Validation set (expert)
val_label_matrix = val_df[LABEL_COLS].values
val_hard_indices = np.argmax(val_label_matrix, axis=1)
val_df['ic_label'] = [class_names[i] for i in val_hard_indices]

# Print class distribution for training
print("\nTraining Set Class Distribution:")
print("-" * 40)
train_class_counts = train_df['ic_label'].value_counts()
for class_name in class_names:
    count = train_class_counts.get(class_name, 0)
    print(f"  {class_name:<15}: {count:5} samples ({100*count/len(train_df):5.1f}%)")
print("-" * 40)

# Print class distribution for validation (expert)
print("\nExpert Validation Set Class Distribution:")
print("-" * 40)
val_class_counts = val_df['ic_label'].value_counts()
for class_name in class_names:
    count = val_class_counts.get(class_name, 0)
    print(f"  {class_name:<15}: {count:5} samples ({100*count/len(val_df):5.1f}%)")
print("-" * 40)

#%% ============================================================
# CREATE DATASETS AND DATALOADERS
# ============================================================
print("\n" + "="*50)
print("CREATING DATASETS AND DATALOADERS")
print("="*50)

# Create separate dataset objects for training and validation
train_dataset = icmobi_dataloader.ICMOBIDatasetFormatter(train_df, transform=transform_pipeline)
val_dataset = icmobi_dataloader.ICMOBIDatasetFormatter(val_df, transform=None)  # No augmentation on expert val set

print(f"Training dataset size: {len(train_dataset)}")
print(f"Expert validation dataset size: {len(val_dataset)}")

# Use distributed dataloading
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

#%% ============================================================
# COMPUTE CLASS WEIGHTS FOR IMBALANCED DATA
# ============================================================
print("\n" + "="*50)
print("COMPUTING CLASS WEIGHTS")
print("="*50)

# Count samples per class in training set
train_labels = train_df['ic_label'].values
train_class_count_arr = []
for class_name in class_names:
    count = np.sum(train_labels == class_name)
    train_class_count_arr.append(max(count, 1))  # Avoid division by zero
    
train_class_count_arr = np.array(train_class_count_arr)

# Compute weights (inverse frequency, normalized)
class_weights = 1.0 / train_class_count_arr
class_weights = class_weights / class_weights.sum() * len(class_names)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class weights (higher = more emphasis on this class):")
for i, class_name in enumerate(class_names):
    print(f"  {class_name:<15}: {class_weights[i].item():.3f} (training samples: {train_class_count_arr[i]})")

#%% ============================================================
# LOAD MODEL
# ============================================================
print("\n" + "="*50)
print("LOADING MODEL")
print("="*50)

rel_path = os.path.abspath("../icmobi_ext/utils/")
model = icmobi_model.ICMoBiNetTransferL(mat_path=os.path.join(rel_path, "netICL.mat"))

# Model to device    
model.to(device)
model = DDP(model)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

#%% ============================================================
# LOSS, OPTIMIZER, SCHEDULER
# ============================================================
# Loss with class weights (handles imbalanced data)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LEARNING_RATE
)

# Learning rate scheduler (reduces LR when validation plateaus)
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='max',           # Maximize validation accuracy
    factor=0.5,           # Reduce LR by half
    patience=5           # Wait 5 epochs before reducing
)

#%% ============================================================
# EARLY STOPPING SETUP
# ============================================================
best_val_acc = 0.0
best_val_loss = float('inf')
best_model_state = None
best_epoch = 0
patience_counter = 0

# Training history tracking
train_history = {'loss': [], 'accuracy': []}
val_history = {'loss': [], 'accuracy': []}

#%% ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "="*50)
print("TRAINING")
print("="*50)
print(f"Max Epochs: {EPOCHS}")
print(f"Early Stopping Patience: {PATIENCE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Training on: {len(train_dataset)} crowdsourced samples")
print(f"Validating on: {len(val_dataset)} expert-labeled samples")
print("="*50)

for epoch in range(EPOCHS):
    train_sampler.set_epoch(epoch)
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)    

    # ---- Training ----
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in tqdm(train_loader, desc="Training"):    
        images = batch['topo'].to(device)
        psds = batch['psd'].to(device)
        acs = batch['ac'].to(device)
        labels = batch['label'].to(device)
        
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

    train_accuracy = train_correct / train_total
    train_epoch_loss = train_loss / train_total
    train_history['loss'].append(train_epoch_loss)
    train_history['accuracy'].append(train_accuracy)
    print(f"Train Loss: {train_epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # ---- Validation (Expert Labels) ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation (Expert)"):
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
    val_epoch_loss = val_loss / val_total
    val_history['loss'].append(val_epoch_loss)
    val_history['accuracy'].append(val_accuracy)
    print(f"Val Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    # ---- Learning Rate Scheduler ----
    scheduler.step(val_accuracy)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning Rate: {current_lr:.6f}")

    # ---- Early Stopping Check ----
    if val_accuracy > best_val_acc + MIN_DELTA:
        best_val_acc = val_accuracy
        best_val_loss = val_epoch_loss
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.module.state_dict())
        patience_counter = 0
        print(f"★ New best validation accuracy: {val_accuracy:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"Best epoch was {best_epoch+1} with val accuracy {best_val_acc:.4f}")
            break

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"Best Epoch: {best_epoch+1}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Best Validation Loss: {best_val_loss:.4f}")

# Load best model
if best_model_state is not None:
    model.module.load_state_dict(best_model_state)
    print("Loaded best model weights.")

#%% ============================================================
# EVALUATION AND EXPORT (Only on rank 0)
# ============================================================
if rank == 0:
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION ON EXPERT LABELS (Best Model)")
    print("="*50)
    
    # Create output directory
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate on expert validation set
    y_true, y_pred, y_probs, metrics = evaluate_model(model, val_loader, device, class_names)
    
    # Generate confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, class_names, 
                               save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Generate ROC curves
    auc_scores = plot_roc_curves(y_true, y_probs, class_names,
                                 save_path=os.path.join(output_dir, 'roc_curves.png'))
    
    # Save training history (with best epoch marker)
    save_model_diagnostics(train_history, val_history, best_epoch=best_epoch,
                          save_path=os.path.join(output_dir, 'training_history.png'))
    
    # Save metrics to CSV
    actual_epochs = len(train_history['loss'])
    metrics_df = pd.DataFrame({
        'Epoch': list(range(1, actual_epochs + 1)),
        'Train_Loss': train_history['loss'],
        'Train_Accuracy': train_history['accuracy'],
        'Val_Loss': val_history['loss'],
        'Val_Accuracy': val_history['accuracy']
    })
    metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)
    print(f"Training metrics saved to {os.path.join(output_dir, 'training_metrics.csv')}")
    
    # Save classification metrics
    class_metrics_data = []
    for i, class_name in enumerate(class_names):
        class_metrics_data.append({
            'Class': class_name,
            'Precision': metrics[i]['precision'],
            'Recall': metrics[i]['recall'],
            'F1-Score': metrics[i]['f1'],
            'Support': int(metrics[i]['support'])
        })
    class_metrics_df = pd.DataFrame(class_metrics_data)
    class_metrics_df.to_csv(os.path.join(output_dir, 'classification_metrics.csv'), index=False)
    print(f"Classification metrics saved to {os.path.join(output_dir, 'classification_metrics.csv')}")
    
    # Save AUC scores
    auc_df = pd.DataFrame({
        'Class': class_names + ['Micro-average'],
        'AUC': [auc_scores[i] for i in range(len(class_names))] + [auc_scores['micro']]
    })
    auc_df.to_csv(os.path.join(output_dir, 'auc_scores.csv'), index=False)
    print(f"AUC scores saved to {os.path.join(output_dir, 'auc_scores.csv')}")
    
    # Save training summary
    summary_df = pd.DataFrame({
        'Metric': ['Best Epoch', 'Best Val Accuracy', 'Best Val Loss', 
                   'Total Epochs Run', 'Random Seed', 'Early Stopping Patience',
                   'Training Samples', 'Expert Validation Samples'],
        'Value': [best_epoch + 1, best_val_acc, best_val_loss, 
                  actual_epochs, RANDOM_SEED, PATIENCE,
                  len(train_dataset), len(val_dataset)]
    })
    summary_df.to_csv(os.path.join(output_dir, 'training_summary.csv'), index=False)
    print(f"Training summary saved to {os.path.join(output_dir, 'training_summary.csv')}")
    
    # Export to ONNX
    sample_batch = next(iter(val_loader))
    input_shapes = {
        'topo': sample_batch['topo'].shape[1:],
        'psd': sample_batch['psd'].shape[1:],
        'ac': sample_batch['ac'].shape[1:]
    }
    
    export_to_onnx(model, 
                   os.path.join(output_dir, 'icmobi_model.onnx'),
                   device,
                   input_shapes)
    
    # Save PyTorch model weights
    torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
               os.path.join(output_dir, 'icmobi_model_weights.pth'))
    print(f"PyTorch weights saved to {os.path.join(output_dir, 'icmobi_model_weights.pth')}")
    
    print("\n" + "="*50)
    print("All outputs saved to:", output_dir)
    print("="*50)
    print("\nCONFIGURATION:")
    print("  ✓ Training: crowdsourced labels (trainset_labeled + trainset_labeled_2)")
    print("  ✓ Validation: expert labels (trainset_labeled_expert)")
    print("  ✓ No augmentation on expert validation set")
    print("  ✓ Class weights computed from training set")
    print("  ✓ Early stopping based on expert validation accuracy")
    print("  ✓ LR scheduler (adaptive learning rate)")
    print("="*50)

cleanup()
