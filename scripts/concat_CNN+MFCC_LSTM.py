# Might consider as our base model due to the lack of exploration of hyperparameters yet

from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["classID"], random_state=42)

# Concatenated Version of CNN-LSTM + MFCC-LSTM

import torch.nn as nn

class DualStreamAudioClassifier(nn.Module):
    def __init__(self, cnn_model, mfcc_model, hidden_dim=128, n_classes=10):
        super().__init__()
        self.cnn_model = cnn_model
        self.mfcc_model = mfcc_model

        # Optionally freeze feature extractors
        for param in self.cnn_model.parameters():
            param.requires_grad = False
        for param in self.mfcc_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, mel_input, mfcc_input):
        mel_feat = self.cnn_model.extract_features(mel_input)    # (B, 128)
        mfcc_feat = self.mfcc_model.extract_features(mfcc_input) # (B, 128)
        merged = torch.cat((mel_feat, mfcc_feat), dim=1)         # (B, 256)
        return self.classifier(merged)


class DualAudioDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mel = torch.tensor(self.df.iloc[idx]["mel"], dtype=torch.float32).unsqueeze(0)  # (1, 128, T)
        mfcc = torch.tensor(self.df.iloc[idx]["mfcc"], dtype=torch.float32).T           # (T, 40)
        label = self.df.iloc[idx]["classID"]
        return mel, mfcc, label


# Prep before training the model
cnn_model = CNNLSTM()  # CNNLSTM is the CNN-based model
mfcc_model = MFCC_LSTM()  # MFCC_LSTM is the MFCC-based LSTM model

# Move models to the appropriate device
cnn_model = cnn_model.to(device)
mfcc_model = mfcc_model.to(device)

# Initialize the dual stream model
dual_model = DualStreamAudioClassifier(cnn_model, mfcc_model).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(dual_model.classifier.parameters(), lr=1e-3)
