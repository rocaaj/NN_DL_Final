class MFCC_LSTM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, n_classes=10, dropout=0.3):
        super(MFCC_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def extract_features(self, x):  # x: (B, T, 40)
        out, _ = self.lstm(x)
        return out[:, -1, :]        # hidden state

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc(features)

def get_lstm_outputs(self, x):  # x: (B, T, 40)
    with torch.no_grad():
        out, _ = self.lstm(x)   # (B, T, H)
    return out
