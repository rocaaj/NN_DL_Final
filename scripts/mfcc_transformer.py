import torch
import torch.nn as nn
import torchaudio

class MFCCTransformerClassifier(nn.Module):
    def __init__(self, n_mfcc=40, num_classes=4, num_layers=2, d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, max_seq_len=200):
        super(MFCCTransformerClassifier, self).__init__()
        self.n_mfcc = n_mfcc
        self.d_model = d_model

        # Linear projection to model dimension
        self.input_proj = nn.Linear(n_mfcc, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch_size, time_steps, n_mfcc]
        x = self.input_proj(x)  # → [batch_size, time_steps, d_model]
        x = self.positional_encoding(x)

        # Transformer expects shape [time_steps, batch_size, d_model]
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)  # → [time_steps, batch_size, d_model]
        x = x.transpose(0, 1)  # → [batch_size, time_steps, d_model]

        # Global pooling
        x = self.global_pool(x.transpose(1, 2)).squeeze(2)  # → [batch_size, d_model]
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
