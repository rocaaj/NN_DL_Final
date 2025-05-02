"""
train_improved_concat_model.py
------------------------------

Description:
    Performs 10-fold cross-validation on UrbanSound8K,
    training a frozen dual‑stream audio classifier (Mel‑CNN + MFCC‑LSTM).
    Records per-fold: macro-F1, average loss, average accuracy, precision, recall.
    Saves best model per fold and plots all five metrics across folds.

How to run:
    python train_improved_concat_model.py

Outputs:
    - best_model_fold{fold}.pth  (for folds 1–10)
    - macro_f1_per_fold.png
    - avg_loss_per_fold.png
    - avg_accuracy_per_fold.png
    - macro_precision_per_fold.png
    - macro_recall_per_fold.png

Author: 
    Osvaldo Hernandez-Segura

References: 
    ChatGPT, Librosa documentation
"""

import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from improved_CNN_LSTM import CNNBranch, LSTMBranch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

# Config & Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR     = os.path.join(os.pardir, 'data')
METADATA_CSV = os.path.join(DATA_DIR, 'UrbanSound8K.csv')
PLOT_OUTPUT_DIR   = os.path.join(os.pardir, 'output', 'improved_concat_model')
N_MFCC       = 40
MAX_LEN      = 200       # time-steps
N_MELS       = 64        # mel bins
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 1e-3
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset: on-the-fly features
class DualAudioDataset(Dataset):
    def __init__(self, df, data_dir, label_map,
                 n_mfcc=N_MFCC, max_len=MAX_LEN, n_mels=N_MELS):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = data_dir
        self.label_map = label_map
        self.n_mfcc    = n_mfcc
        self.max_len   = max_len
        self.n_mels    = n_mels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        fold  = row['fold']
        fname = row['slice_file_name']
        label = self.label_map[row['class']]
        path  = os.path.join(self.data_dir, f"fold{fold}", fname)

        y, sr = librosa.load(path, sr=None)
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc).T
        if mfcc.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - mfcc.shape[0], self.n_mfcc))
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:self.max_len, :]

        # Mel-spectrogram + dB + delta
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < self.max_len:
            pad = np.zeros((self.n_mels, self.max_len - mel_db.shape[1]))
            mel_db = np.hstack([mel_db, pad])
        else:
            mel_db = mel_db[:, :self.max_len]
        mel_dlt = librosa.feature.delta(mel_db)
        mel = np.stack([mel_db, mel_dlt], axis=0)

        return (torch.tensor(mel, dtype=torch.float32),
                torch.tensor(mfcc, dtype=torch.float32),
                label)

# Fusion model
class DualStreamAudioClassifier(nn.Module):
    def __init__(self, cnn: CNNBranch, lstm: LSTMBranch, n_classes):
        super().__init__()
        self.cnn  = cnn
        self.lstm = lstm

        # freeze
        for p in self.cnn.parameters(): p.requires_grad = False
        for p in self.lstm.parameters(): p.requires_grad = False

        # fusion head
        self.fusion_dense = nn.Linear(128 + 25, 128)
        self.lstm_head    = nn.LSTM(128, 64, batch_first=True)
        self.classifier   = nn.Linear(64, n_classes)

    def forward(self, mel, mfcc):
        f1 = self.cnn.extract_features(mel)    # (B,128)
        f2 = self.lstm.extract_features(mfcc)  # (B,25)
        x  = torch.cat([f1, f2], dim=1)        # (B,153)
        x  = self.fusion_dense(x)              # (B,128)
        x  = x.unsqueeze(1)                    # (B,1,128)
        o2,_ = self.lstm_head(x)               # (B,1,64)
        o2    = o2[:,0,:]                      # (B,64)
        return self.classifier(o2)            # logits

# Train helper
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for mel, mfcc, lbl in loader:
        mel,mfcc,lbl = mel.to(DEVICE), mfcc.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        logits = model(mel, mfcc)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
        correct    += (logits.argmax(1) == lbl).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

# Plot helper
def plot_metric(folds, vals, ylabel, title, fname, color, marker):
    plt.figure()
    plt.plot(folds, vals, color=color, marker=marker, linewidth=2, markersize=8)
    plt.title(title)
    plt.xlabel('Fold')
    plt.ylabel(ylabel)
    plt.xticks(folds)
    plt.grid(True)
    plt.savefig(fname)
    plt.show()
    plt.close()

# Main procedure
def main():
    print(f"Using device: {DEVICE}")

    df = pd.read_csv(METADATA_CSV)
    folds = list(range(1, 11))
    classes = sorted(df['class'].unique())
    label_map = {c:i for i,c in enumerate(classes)}
    n_classes = len(classes)

    losses, accs, f1s, precs, recs = [], [], [], [], []

    for fold in folds:
        train_df = df[df['fold'] != fold]
        val_df   = df[df['fold'] == fold]

        train_loader = DataLoader(DualAudioDataset(train_df, DATA_DIR, label_map),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(DualAudioDataset(val_df,   DATA_DIR, label_map),
                                  batch_size=BATCH_SIZE, shuffle=False)

        cnn_model  = CNNBranch().to(DEVICE)
        lstm_model = LSTMBranch(input_dim=N_MFCC).to(DEVICE)
        model      = DualStreamAudioClassifier(cnn_model, lstm_model, n_classes).to(DEVICE)
        optimizer  = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        criterion  = nn.CrossEntropyLoss()

        best_loss = float('inf')
        for _ in range(EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion)

            # validation for checkpoint
            val_loss, _   = train_one_epoch(model, val_loader, optimizer, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f'best_model_fold{fold}.pth')

        # load best and compute metrics
        model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
        model.eval()

        all_preds, all_lbls = [], []
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for mel,mfcc,lbl in val_loader:
                mel,mfcc,lbl = mel.to(DEVICE), mfcc.to(DEVICE), lbl.to(DEVICE)
                logits = model(mel, mfcc)
                val_loss += criterion(logits,lbl).item() * mel.size(0)
                preds = logits.argmax(1)
                correct += (preds == lbl).sum().item()
                all_preds.append(preds.cpu().numpy())
                all_lbls.append(lbl.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc  = correct / len(val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_lbls  = np.concatenate(all_lbls)

        f1   = f1_score(all_lbls, all_preds, average='macro')
        prec = precision_score(all_lbls, all_preds, average='macro', zero_division=0)
        rec  = recall_score(all_lbls, all_preds, average='macro', zero_division=0)

        losses.append(val_loss)
        accs.append(val_acc)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
    
    # Plot all metrics
    plot_metric(folds, f1s,   'Macro F1-score',    'Macro F1-score per Fold',    os.path.join(PLOT_OUTPUT_DIR, 'macro_f1_per_fold.png'),    'green', 's')
    plot_metric(folds, losses,'Loss',               'Average Loss per Fold',      os.path.join(PLOT_OUTPUT_DIR, 'avg_loss_per_fold.png'),    'blue',  'o')
    plot_metric(folds, accs,  'Accuracy',           'Average Accuracy per Fold',  os.path.join(PLOT_OUTPUT_DIR, 'avg_accuracy_per_fold.png'),'orange','^')
    plot_metric(folds, precs,'Macro Precision',     'Macro Precision per Fold',   os.path.join(PLOT_OUTPUT_DIR, 'macro_precision_per_fold.png'),'purple','D')
    plot_metric(folds, recs, 'Macro Recall',        'Macro Recall per Fold',      os.path.join(PLOT_OUTPUT_DIR, 'macro_recall_per_fold.png'),   'red',   'v')

if __name__ == '__main__':
    main()
