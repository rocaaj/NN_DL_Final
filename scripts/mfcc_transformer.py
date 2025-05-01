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
        )

        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, 128))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, T, C) => (B, C, T)
        x = x.transpose(1, 2)
        x = self.cnn(x)               # (B, 128, T)
        x = x.transpose(1, 2)         # (B, T, 128)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)       # (B, T, 128)

        attn_weights = self.att_pool(x)       # (B, T, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (B, 128)

        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
