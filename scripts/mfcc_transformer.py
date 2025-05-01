"""
CNNTransformerClassifier (PyTorch)

Author: Anthony Roca
--------------------

This module defines a hybrid CNN + Transformer architecture for audio classification 
using precomputed MFCC features. It is optimized for the UrbanSound8K dataset and 
designed to capture both local and global temporal patterns in time-series MFCC inputs.

Architecture Summary:
---------------------
- CNN stack: 3 1D convolutional layers for local feature extraction
- Positional encoding: Learnable embeddings to inject temporal order
- Transformer encoder: 4-layer transformer with 8 heads and FFN size 512
- Attention pooling: Learns temporal weights for aggregating sequence features
- Final classifier: LayerNorm, dropout, linear output for 10-class classification

Optimized for GPU usage and integration with training scripts that use mixup, 
focal loss, label smoothing, and 10-fold cross-validation.

Input shape:  (batch_size, sequence_length, n_mfcc)
Output shape: (batch_size, num_classes)

Acknowledgment:
---------------
This implementation benefited from iterative debugging and generalization guidance provided by OpenAI's ChatGPT.
"""


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
