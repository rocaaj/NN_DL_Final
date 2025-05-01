import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTransformerClassifier(nn.Module):
    def __init__(self, n_mfcc, num_classes, max_seq_len):
        super(CNNTransformerClassifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, 256))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.att_pool = nn.Sequential(
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, T, C) => (B, C, T)
        x = x.transpose(1, 2)
        x = self.cnn(x)               # (B, 256, T)
        x = x.transpose(1, 2)         # (B, T, 256)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)       # (B, T, 256)

        attn_weights = self.att_pool(x)         # (B, T, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (B, 256)

        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
