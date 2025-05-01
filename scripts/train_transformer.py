import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import numpy as np
from mfcc_transformer import MFCCTransformerClassifier
import random
from collections import Counter

# -----------------------------
# Config
# -----------------------------

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = '../data/'
METADATA_CSV = '../data/UrbanSound8K.csv'
N_MFCC = 40
MAX_LEN = 200
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
PATIENCE = 5

USE_CLASS_WEIGHTS = True
USE_NORMALIZED_MFCC = True
USE_WEIGHTED_SAMPLER = True
USE_AUGMENTATION = True
USE_DROPOUT = True
USE_LABEL_SMOOTHING = True
USE_MIXUP = True

cuda_id = 0
if torch.cuda.is_available():
    torch.cuda.set_device(cuda_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Mixup Helper
# -----------------------------
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -----------------------------
# EarlyStopping Helper
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
# Dataset
# -----------------------------
class MFCCDataset(Dataset):
    def __init__(self, df, data_dir, label_map, n_mfcc=40, max_len=200, training=True):
        self.data_dir = data_dir
        self.df = df
        self.label_map = label_map
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, f"fold{row['fold']}", row['filename'])
        label = self.label_map[row['label']]

        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc).T

        if USE_NORMALIZED_MFCC:
            mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-6)

        if USE_AUGMENTATION and self.training:
            if random.random() < 0.3:
                mfcc += np.random.normal(0, 0.05, mfcc.shape)
            if random.random() < 0.3 and mfcc.shape[0] > 10:
                t = random.randint(0, mfcc.shape[0] - 10)
                mfcc[t:t+10] = 0

        if mfcc.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - mfcc.shape[0], self.n_mfcc))
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:self.max_len, :]

        return torch.tensor(mfcc, dtype=torch.float32), label

# -----------------------------
# Training / Evaluation
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
# Output Utilities
# -----------------------------
def save_metrics_and_plots(metrics_df, class_names):
    metrics_df.to_csv("cv_metrics_report.csv", index=False)

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
# Model + Optimizer Setup
# -----------------------------
def create_model_and_optimizer(label_map, label_counts):
    model = MFCCTransformerClassifier(n_mfcc=N_MFCC, num_classes=len(label_map), max_seq_len=MAX_LEN)
    if USE_DROPOUT:
        model.dropout = nn.Dropout(0.3)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if USE_CLASS_WEIGHTS:
        class_weights = [1.0 / label_counts[i] for i in range(len(label_map))]
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1 if USE_LABEL_SMOOTHING else 0.0)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if USE_LABEL_SMOOTHING else 0.0)
    return model, optimizer, scheduler, criterion

# -----------------------------
# Fold Training Logic
# -----------------------------
def run_fold(fold, df, label_map):
    print(f"\nFold {fold}/10")
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)

    train_dataset = MFCCDataset(train_df, DATA_DIR, label_map, N_MFCC, MAX_LEN, training=True)
    val_dataset = MFCCDataset(val_df, DATA_DIR, label_map, N_MFCC, MAX_LEN, training=False)

    if USE_WEIGHTED_SAMPLER:
        labels = train_df['label'].map(label_map).tolist()
        label_counts = np.bincount(labels)
        weights = 1. / label_counts
        sample_weights = [weights[label] for label in labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    label_counts = np.bincount(train_df['label'].map(label_map).tolist())

    model, optimizer, scheduler, criterion = create_model_and_optimizer(label_map, label_counts)
    early_stopper = EarlyStopping(patience=PATIENCE)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        y_true, y_pred = evaluate(model, val_loader)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f} - Macro F1: {f1:.4f}")
        if early_stopper.step(f1):
            print("  Early stopping triggered")
            break

    y_true, y_pred = evaluate(model, val_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    _, _, per_class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    print(f"  Fold {fold} → Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    return loss, f1, per_class_f1

# -----------------------------
# Main Execution
# -----------------------------
def main():
    df = pd.read_csv(METADATA_CSV)
    df = df.rename(columns={'slice_file_name': 'filename', 'class': 'label'})
    class_names = sorted(df['label'].unique())
    label_map = {label: i for i, label in enumerate(class_names)}

    fold_losses = []
    fold_macro_f1s = []
    fold_per_class_f1s = []

    for fold in range(1, 11):
        loss, f1, per_class_f1 = run_fold(fold, df, label_map)
        fold_losses.append(loss)
        fold_macro_f1s.append(f1)
        fold_per_class_f1s.append(per_class_f1)

    metrics_df = pd.DataFrame({
        'fold': list(range(1, 11)),
        'avg_loss': fold_losses,
        'macro_f1': fold_macro_f1s
    })
    for i, class_name in enumerate(class_names):
        metrics_df[class_name] = [f[i] for f in fold_per_class_f1s]

    save_metrics_and_plots(metrics_df, class_names)

if __name__ == "__main__":
    main()
