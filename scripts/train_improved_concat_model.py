"""
Description:
    This project builds and trains a dual‑stream audio classifier that fuses Mel‑spectrogram 
    and MFCC features. A frozen CNN processes Mel‑spectrogram inputs; a frozen two‑layer LSTM 
    processes MFCC sequences. Their embeddings are concatenated, passed through a small dense→LSTM 
    fusion head, then classified over 10 classes.

How to run: 
    python train_improved_concat_model.py

------------------------------

Imports the CNN/LSTM modules, builds fusion head, trains only that head.
"""

import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from improved_CNN_LSTM import CNNBranch, LSTMBranch

# -----------------------------
# Config & Seeds
# -----------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR       = '../data/'
METADATA_CSV   = '../data/UrbanSound8K.csv'
N_MFCC         = 40
MAX_LEN        = 200       # number of time-steps for MFCC and mel
N_MELS         = 64        # number of mel bins
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 1e-3

# -----------------------------
# Dataset: compute mel + mfcc on the fly
# -----------------------------
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
        row      = self.df.iloc[idx]
        fold     = row['fold']
        fname    = row['slice_file_name']
        label    = self.label_map[row['class']]
        path     = os.path.join(self.data_dir, f"fold{fold}", fname)

        # load waveform
        y, sr = librosa.load(path, sr=None)

        # MFCC  → shape (time, n_mfcc)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc).T
        if mfcc.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - mfcc.shape[0], self.n_mfcc))
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:self.max_len, :]

        # # Mel‑spectrogram + delta → 2 channels, shape (2, n_mels, time)
        # mel      = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        # mel_db   = librosa.power_to_db(mel, ref=np.max)
        # mel_dlt  = librosa.feature.delta(mel_db)
        # # pad/truncate time-axis to max_len
        # T = mel_db.shape[1]
        # if T < self.max_len:
        #     pad = np.zeros((self.n_mels, self.max_len - T))
        #     mel_db  = np.hstack([mel_db,  pad])
        #     mel_dlt = np.hstack([mel_dlt, pad])
        # else:
        #     mel_db  = mel_db[:, :self.max_len]
        #     mel_dlt = mel_dlt[:, :self.max_len]

        # mel = np.stack([mel_db, mel_dlt], axis=0)

        # # to tensors
        # mel_tensor  = torch.tensor(mel,  dtype=torch.float32)
        # mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        # return mel_tensor, mfcc_tensor, label

        # Mel‑spectrogram → dB, pad/truncate to max_len, then compute delta
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # pad or truncate time‐axis to self.max_len
        if mel_db.shape[1] < self.max_len:
            pad = np.zeros((self.n_mels, self.max_len - mel_db.shape[1]))
            mel_db = np.hstack([mel_db, pad])
        else:
            mel_db = mel_db[:, :self.max_len]
        # now compute delta on fixed‐length mel_db
        mel_dlt = librosa.feature.delta(mel_db)
        mel = np.stack([mel_db, mel_dlt], axis=0)

        # to tensors
        mel_tensor  = torch.tensor(mel,  dtype=torch.float32)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        return mel_tensor, mfcc_tensor, label


# -----------------------------
# Model, Train & Eval
# -----------------------------
class DualStreamAudioClassifier(nn.Module):
    def __init__(self, cnn: CNNBranch, lstm: LSTMBranch, n_classes=10):
        super().__init__()
        self.cnn  = cnn
        self.lstm = lstm
        # freeze pre-trained branches
        for p in self.cnn.parameters():  p.requires_grad = False
        for p in self.lstm.parameters(): p.requires_grad = False

        # fusion head: 80+25→128→LSTM(128→64)→10
        self.fusion_dense = nn.Linear(80 + 25, 128)
        self.lstm_head    = nn.LSTM(128, 64, batch_first=True)
        self.classifier   = nn.Linear(64, n_classes)

    def forward(self, mel, mfcc):
        f1     = self.cnn.extract_features(mel)    # (B,80)
        f2     = self.lstm.extract_features(mfcc)  # (B,25)
        x      = torch.cat([f1, f2], dim=1)        # (B,105)
        x      = self.fusion_dense(x)              # (B,128)
        x      = x.unsqueeze(1)                    # (B,1,128)
        o2, _  = self.lstm_head(x)                 # (B,1,64)
        o2     = o2[:, 0, :]                       # (B,64)
        logits = self.classifier(o2)               # (B,10)
        return logits                              # raw logits

def train_epoch(model, loader, opt, crit, dev):
    model.train()
    loss_sum, corr = 0.0, 0
    for mel, mfcc, lbl in loader:
        mel, mfcc, lbl = mel.to(dev), mfcc.to(dev), lbl.to(dev)
        opt.zero_grad()
        logits = model(mel, mfcc)
        loss   = crit(logits, lbl)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * mel.size(0)
        corr     += (logits.argmax(1) == lbl).sum().item()
    return loss_sum / len(loader.dataset), corr / len(loader.dataset)

def evaluate(model, loader, crit, dev):
    model.eval()
    loss_sum, corr = 0.0, 0
    with torch.no_grad():
        for mel, mfcc, lbl in loader:
            mel, mfcc, lbl = mel.to(dev), mfcc.to(dev), lbl.to(dev)
            logits = model(mel, mfcc)
            loss   = crit(logits, lbl)
            loss_sum += loss.item() * mel.size(0)
            corr     += (logits.argmax(1) == lbl).sum().item()
    return loss_sum / len(loader.dataset), corr / len(loader.dataset)

# -----------------------------
# Main
# -----------------------------
def main():
    
    # gpu setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device(),
            torch.cuda.get_device_name(torch.cuda.current_device()))

    # load metadata & train/val split
    df = pd.read_csv(METADATA_CSV)
    df = df.rename(columns={'slice_file_name':'slice_file_name','class':'class'})
    # stratify by class label
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['class'], random_state=42
    )
    # label mapping
    classes   = sorted(df['class'].unique())
    label_map = {c:i for i,c in enumerate(classes)}

    # loaders
    train_loader = DataLoader(
        DualAudioDataset(train_df, DATA_DIR, label_map),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        DualAudioDataset(val_df, DATA_DIR, label_map),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # model
    cnn_model    = CNNBranch().to(device)
    lstm_model   = LSTMBranch(input_dim=N_MFCC).to(device)
    fusion_model = DualStreamAudioClassifier(
        cnn_model, lstm_model, n_classes=len(classes)
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, fusion_model.parameters()),
        lr=LR
    )
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    for epoch in range(1, EPOCHS+1):
        tr_l, tr_a = train_epoch(fusion_model, train_loader,
                                 optimizer, criterion, device)
        vl_l, vl_a = evaluate(fusion_model, val_loader,
                               criterion, device)
        print(f"Epoch {epoch:02d} | Train Acc: {tr_a:.4f}, Val Acc: {vl_a:.4f}")
        if vl_l < best_loss:
            best_loss = vl_l
            torch.save(fusion_model.state_dict(),
                       "best_improved_CNN_LSTM_model.pth")

if __name__ == "__main__":
    main()
