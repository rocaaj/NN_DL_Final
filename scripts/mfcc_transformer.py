import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# -----------------------------
# Transformer-based Classifier for MFCCs
# -----------------------------
class MFCCTransformerClassifier(nn.Module):
    def __init__(self, n_mfcc, num_classes, max_seq_len):
        super(MFCCTransformerClassifier, self).__init__()
        self.n_mfcc = n_mfcc
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(n_mfcc, 128)
        self.pos_encoder = PositionalEncoding(d_model=128, max_len=max_seq_len, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.norm(x[:, 0, :])
        x = self.dropout(x)
        return self.classifier(x)
