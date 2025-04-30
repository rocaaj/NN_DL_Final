import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, n_mels=128, time_steps=174, lstm_hidden=128, n_classes=10):
        super(CNNLSTM, self).__init__()

        # Four convolutional blocks
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # -> (16, 128, 174)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (16, 64, 87)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (32, 64, 87)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (32, 32, 43)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (64, 32, 43)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (64, 16, 21)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> (128, 16, 21)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # -> (128, 8, 10)
        )

        # LSTM layer
        self.lstm_input_dim = 128 * 8  # features per time step (channels × height)
        self.lstm_seq_len = 10         # width becomes time steps
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim,
                            hidden_size=lstm_hidden,
                            num_layers=1,
                            batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):  # x: (B, 1, 128, 174)
        x = self.conv_block(x)         # -> (B, 128, 8, 10)
        x = x.permute(0, 3, 1, 2)      # -> (B, 10, 128, 8) : time-major
        x = x.reshape(x.size(0), x.size(1), -1)  # -> (B, 10, 128×8 = 1024)

        lstm_out, _ = self.lstm(x)     # -> (B, 10, hidden)
        out = lstm_out[:, -1, :]       # take last time step → (B, hidden)
        out = self.fc(out)             # -> (B, n_classes)
        return out
