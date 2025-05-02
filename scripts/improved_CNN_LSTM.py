"""
Improved CNN-LSTM model for audio classification.
Used by the train_improved_concat_model.py script.

Author: 
    Osvaldo Hernandez-Segura
References:
    ChatGPT
"""
import torch
import torch.nn as nn

class CNNBranch(nn.Module):
    """
    Mel‐spectrogram branch as in Fig. 6: 4×Conv–BN–ReLU–Pool blocks → 128‐d embedding.
    """
    def __init__(self):
        super().__init__()
        # Block 1: in=2,   out=16
        self.conv1 = nn.Sequential(
            nn.Conv2d(2,  16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Block 2: 16 → 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Block 3: 32 → 64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Block 4: 64 → 128 + global pooling
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1,1))
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.view(x.size(0), -1)  # → (batch, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)


class LSTMBranch(nn.Module):
    """
    MFCC branch (unchanged):
      - LSTM(40→108)→dropout
      - LSTM(108→50)→dropout
      - Dense(50→25) + dropout
    """
    def __init__(self, input_dim=40, h1=108, h2=50, d1=25, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, h1, batch_first=True, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(h1,      h2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(h2, d1)
        self.relu  = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(dropout)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        o1, _ = self.lstm1(x)
        o1     = self.drop1(o1)
        o2, _ = self.lstm2(o1)
        h2     = o2[:, -1, :]
        h2     = self.drop2(h2)
        h3     = self.relu(self.fc1(h2))
        return self.drop3(h3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)
