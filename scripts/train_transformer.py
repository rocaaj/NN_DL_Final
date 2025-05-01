"""
train_transformer.py

Author: Anthony Roca
--------------------

This script trains a hybrid CNN + Transformer model for environmental sound classification
on the UrbanSound8K dataset using 10-fold cross-validation. It is optimized for GPU usage
and integrates several advanced training strategies to improve generalization and robustness.

Key Features:
-------------
- Model: CNN backbone for local feature extraction + Transformer encoder for global context
- Input: Precomputed MFCC tensors stored in ../data/mfcc_cache/
- Loss Function: Focal loss for addressing class imbalance
- Regularization: Mixup data augmentation, label smoothing, dropout
- Evaluation: Macro precision, recall, and F1-score reported per fold
- Optimizations: Weighted sampling, early stopping, cosine learning rate scheduler
- GPU Acceleration: Utilizes PyTorch DataLoader with multiple workers and persistent loading

Expected Input Shape: (batch_size, time_steps, n_mfcc)
Output: Class probability logits for 10 sound categories

Acknowledgment:
---------------
This implementation benefited from iterative debugging and generalization guidance provided by OpenAI's ChatGPT.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mfcc_transformer import CNNTransformerClassifier
import random

# -----------------------------
# Config: Set hyperparameters, file paths, and training flags
# -----------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True  # Enables autotuner for faster GPU conv

# Define global constants
DATA_DIR         = '../data/'
MFCC_CACHE_DIR   = '../data/mfcc_cache/'
METADATA_CSV     = '../data/UrbanSound8K.csv'
N_MFCC           = 40
MAX_LEN          = 200
BATCH_SIZE       = 64
EPOCHS           = 50
LR               = 1e-4
PATIENCE         = 10

# Training flags
USE_DROPOUT          = True
USE_LABEL_SMOOTHING  = True
USE_MIXUP            = True
USE_WEIGHTED_SAMPLER = True

# DataLoader settings
NUM_WORKERS_TRAIN = 16
NUM_WORKERS_VAL   = 8
PREFETCH_FACTOR   = 2

# -----------------------------
# Device setup: configure for CUDA
# -----------------------------
cuda_id = 0
assert torch.cuda.is_available(), "CUDA required but not available"
torch.cuda.set_device(cuda_id)
device = torch.device(f'cuda:{cuda_id}')
print(f"Running on {torch.cuda.get_device_name(cuda_id)}")

# -----------------------------
# Mixup Helper: interpolates samples for regularization
# -----------------------------
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device) # Beta distribution
    else:
        lam = torch.tensor(1.0, device=x.device) # No mixup
    
    # Ensure lam is in the same dtype as x
    lam = lam.to(dtype=x.dtype)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]

    return mixed_x, y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -----------------------------
# EarlyStopping: stops training when metric stalls
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=PATIENCE):
        self.patience = patience
        self.best_score = None
        self.counter = 0

    def step(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

# -----------------------------
# Focal Loss: down-weights well-classified examples
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none') # CrossEntropyLoss without reduction

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss # Focal loss formula

        if self.reduction == 'mean':
            return focal_loss.mean() # Mean reduction
        elif self.reduction == 'sum':
            return focal_loss.sum() # Sum reduction
        
        return focal_loss

# -----------------------------
# Dataset: Loads cached MFCC tensors and labels
# -----------------------------
class MFCCDataset(Dataset):
    def __init__(self, df, label_map, cache_dir):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    # Load MFCC tensor and label
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fold = row['fold']
        filename = row['filename']
        label = self.label_map[row['label']]

        mfcc_path = os.path.join(self.cache_dir, f"{fold}_{filename}.pt") # Path to precomputed MFCC
        mfcc = torch.load(mfcc_path)

        return mfcc, torch.tensor(label, dtype=torch.long)

# -----------------------------
# Train: forward pass + loss + backward pass
# -----------------------------
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if USE_MIXUP:
            x, y_a, y_b, lam = mixup_data(x, y)
            output = model(x)
            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        else:
            output = model(x)
            loss = criterion(output, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# -----------------------------
# Eval: collects predictions on validation set
# -----------------------------
def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            output = model(x)
            preds = output.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y)
    return y_true, y_pred

# -----------------------------
# Save plots and CSV of metrics
# -----------------------------
def save_metrics_and_plots(metrics_df, class_names):
    metrics_df.to_csv("cv_metrics_report.csv", index=False)

    # Loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(metrics_df['fold'], metrics_df['avg_loss'], marker='o', label='Avg Loss')
    plt.title("Average Loss per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("avg_loss_per_fold.png")
    plt.show()

    # F1 plot
    plt.figure(figsize=(8, 4))
    plt.plot(metrics_df['fold'], metrics_df['macro_f1'], marker='s', color='green', label='Macro F1')
    plt.title("Macro F1-score per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Macro F1-score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("macro_f1_per_fold.png")
    plt.show()

# -----------------------------
# Model and training utility setup
# -----------------------------
def create_model_and_optimizer(label_map, label_counts):
    model = CNNTransformerClassifier(n_mfcc=N_MFCC, num_classes=len(label_map), max_seq_len=MAX_LEN)

    if USE_DROPOUT:
        model.dropout = nn.Dropout(0.3)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    return model, optimizer, scheduler, criterion

# -----------------------------
# Fold-wise training + early stopping
# -----------------------------
def run_fold(fold, df, label_map):
    print(f"\nFold {fold}/10")
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)

    train_dataset = MFCCDataset(train_df, label_map, cache_dir=MFCC_CACHE_DIR)
    val_dataset = MFCCDataset(val_df, label_map, cache_dir=MFCC_CACHE_DIR)

    labels = train_df['label'].map(label_map).tolist()
    label_counts = np.bincount(labels)

    # Use weighted sampling if enabled to balance classes
    if USE_WEIGHTED_SAMPLER:
        weights = 1. / label_counts
        sample_weights = [weights[label] for label in labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=NUM_WORKERS_TRAIN, pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS_TRAIN, pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)

    # Validation DataLoader (no shuffling)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS_VAL, pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)

    # Initialize model, optimizer, scheduler, and loss function
    model, optimizer, scheduler, criterion = create_model_and_optimizer(label_map, label_counts)
    early_stopper = EarlyStopping(patience=PATIENCE)

    # Training loop
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        y_true, y_pred = evaluate(model, val_loader)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print(f"  Epoch {epoch}/{EPOCHS} - Loss: {loss:.4f} - Macro F1: {f1:.4f}")
        if early_stopper.step(f1):
            print("  Early stopping triggered")
            break

    # Final evaluation
    y_true, y_pred = evaluate(model, val_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    _, _, per_class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    print(f"  Fold {fold} → Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    return loss, f1, per_class_f1

# -----------------------------
# Entrypoint: iterate over folds and compile metrics
# -----------------------------
def main():
    # Load metadata and create label map
    df = pd.read_csv(METADATA_CSV)
    df = df.rename(columns={'slice_file_name': 'filename', 'class': 'label'})
    class_names = sorted(df['label'].unique())
    label_map = {label: i for i, label in enumerate(class_names)}

    # Initialize metrics storage
    fold_losses = []
    fold_macro_f1s = []
    fold_per_class_f1s = []

    # Iterate over each fold for training and evaluation
    for fold in range(1, 11):
        loss, f1, per_class_f1 = run_fold(fold, df, label_map)
        fold_losses.append(loss)
        fold_macro_f1s.append(f1)
        fold_per_class_f1s.append(per_class_f1)

    # Compile metrics into a DataFrame
    metrics_df = pd.DataFrame({
        'fold': list(range(1, 11)),
        'avg_loss': fold_losses,
        'macro_f1': fold_macro_f1s
    })
    
    # Add per-class F1 scores to the DataFrame
    for i, class_name in enumerate(class_names):
        metrics_df[class_name] = [f[i] for f in fold_per_class_f1s]

    # Save metrics and plots
    save_metrics_and_plots(metrics_df, class_names)

if __name__ == "__main__":
    main()
