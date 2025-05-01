import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import numpy as np
from mfcc_transformer import MFCCTransformerClassifier
import random

# -----------------------------
# Config
# -----------------------------

# set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = 'data/'
METADATA_CSV = 'metadata.csv'
N_MFCC = 40
MAX_LEN = 200
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

# -----------------------------
# Dataset
# -----------------------------
class MFCCDataset(Dataset):
    def __init__(self, df, data_dir, label_map, n_mfcc=40, max_len=200):
        self.data_dir = data_dir
        self.df = df
        self.label_map = label_map
        self.n_mfcc = n_mfcc
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, f"fold{row['fold']}", row['filename'])
        label = self.label_map[row['label']]

        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc).T

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
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, torch.tensor(y))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            preds = output.argmax(dim=1).numpy()
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
def create_model_and_optimizer(label_map):
    model = MFCCTransformerClassifier(n_mfcc=N_MFCC, num_classes=len(label_map), max_seq_len=MAX_LEN)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

# -----------------------------
# Fold Training Logic
# -----------------------------
def run_fold(fold, df, label_map):
    print(f"\nFold {fold}/10")
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)

    train_loader = DataLoader(MFCCDataset(train_df, DATA_DIR, label_map, N_MFCC, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MFCCDataset(val_df, DATA_DIR, label_map, N_MFCC, MAX_LEN), batch_size=BATCH_SIZE)

    model, optimizer, criterion = create_model_and_optimizer(label_map)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

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
