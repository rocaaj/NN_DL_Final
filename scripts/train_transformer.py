import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import librosa
import numpy as np
from model import MFCCTransformerClassifier  # assumes your model is in model.py

# -----------------------------
# Config
# -----------------------------
DATA_DIR = 'data/'  # Folder with .wav files
LABELS = {'drill': 0, 'hammer': 1, 'jigsaw': 2, 'screwdriver': 3}  # Adjust to your dataset
N_MFCC = 40
MAX_LEN = 200
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4

# -----------------------------
# Dataset
# -----------------------------
class MFCCDataset(Dataset):
    def __init__(self, root_dir, label_map, n_mfcc=40, max_len=200):
        self.filepaths = []
        self.labels = []
        self.n_mfcc = n_mfcc
        self.max_len = max_len

        for label_name, label_id in label_map.items():
            folder = os.path.join(root_dir, label_name)
            for file in os.listdir(folder):
                if file.endswith('.wav'):
                    self.filepaths.append(os.path.join(folder, file))
                    self.labels.append(label_id)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]

        y, sr = librosa.load(path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc).T

        # Pad or truncate
        if mfcc.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - mfcc.shape[0], self.n_mfcc))
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:self.max_len, :]

        return torch.tensor(mfcc, dtype=torch.float32), label

# -----------------------------
# Train and Evaluate
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
    print(classification_report(y_true, y_pred, target_names=LABELS.keys()))

# -----------------------------
# Main
# -----------------------------
def main():
    dataset = MFCCDataset(DATA_DIR, LABELS, n_mfcc=N_MFCC, max_len=MAX_LEN)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = MFCCTransformerClassifier(n_mfcc=N_MFCC, num_classes=len(LABELS), max_seq_len=MAX_LEN)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")
        evaluate(model, val_loader)

if __name__ == "__main__":
    main()
