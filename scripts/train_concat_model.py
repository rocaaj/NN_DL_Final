device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dual_model = DualStreamAudioClassifier(cnn_model, mfcc_model).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(dual_model.classifier.parameters(), lr=1e-3)

def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for mel, mfcc, label in loader:
        mel, mfcc, label = mel.to(device), mfcc.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(mel, mfcc)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
        correct += (output.argmax(1) == label).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for mel, mfcc, label in loader:
            mel, mfcc, label = mel.to(device), mfcc.to(device), label.to(device)
            output = model(mel, mfcc)
            loss = criterion(output, label)
            total_loss += loss.item() * mel.size(0)
            correct += (output.argmax(1) == label).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


train_loader = DataLoader(DualAudioDataset(df_train), batch_size=32, shuffle=True)
val_loader = DataLoader(DualAudioDataset(df_val), batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(dual_model.classifier.parameters(), lr=1e-3)


for epoch in range(1, 11):
    train_loss, train_acc = train_epoch(dual_model, train_loader)
    val_loss, val_acc = evaluate(dual_model, val_loader)
    print(f"Epoch {epoch:02d}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")


# Initial Baseline Performance:
  
# Epoch 01: Train Acc = 0.1749, Val Acc = 0.2215
# Epoch 02: Train Acc = 0.2448, Val Acc = 0.2358
# Epoch 03: Train Acc = 0.2732, Val Acc = 0.2793
# Epoch 04: Train Acc = 0.2776, Val Acc = 0.2988
# Epoch 05: Train Acc = 0.3001, Val Acc = 0.3160
# Epoch 06: Train Acc = 0.3074, Val Acc = 0.3211
# Epoch 07: Train Acc = 0.3147, Val Acc = 0.3349
# Epoch 08: Train Acc = 0.3307, Val Acc = 0.3194
# Epoch 09: Train Acc = 0.3307, Val Acc = 0.3303
# Epoch 10: Train Acc = 0.3404, Val Acc = 0.3515
