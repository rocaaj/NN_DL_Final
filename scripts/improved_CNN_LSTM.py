import torch
import torch.nn as nn

class CNNBranch(nn.Module):
    """Mel-spectrogram branch: 2→64→80 conv + global pooling → 80-dim."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2,  64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                    # → (64,30,20)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1,1))        # → (80,1,1)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x.view(x.size(0), -1)           # (B,80)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)


class LSTMBranch(nn.Module):
    """
    MFCC branch:
      • LSTM(60→108)→dropout
      • LSTM(108→50)→dropout
      • Dense(50→25) + dropout
    """
    def __init__(self, input_dim=60, h1=108, h2=50, d1=25, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, h1, batch_first=True, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(h1,      h2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(h2, d1)
        self.relu  = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(dropout)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        o1, _ = self.lstm1(x)                # → (B, T=41, 108)
        o1     = self.drop1(o1)
        o2, _ = self.lstm2(o1)               # → (B, T=41, 50)
        h2     = o2[:, -1, :]                # last step → (B,50)
        h2     = self.drop2(h2)
        h3     = self.relu(self.fc1(h2))     # → (B,25)
        return self.drop3(h3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)
