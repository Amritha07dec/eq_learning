# train_model.py
from train import (
    pad_samples, split_data, TimeSeriesDataset,
    LSTMClassifier, train, Conv1DLSTMClassifier,
    Attention_over_timestep_LSTMClassifier,
    Attention_over_features_LSTMClassifier,
    set_seed
)
from data_preprocessing.delete import time_series_list, labels
import numpy as np
import torch
import torch.nn as nn

import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# === Reproducibility ===
seed = 123
set_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Data Prep ===
padded_samples = pad_samples(time_series_list, target_dim=6)
X_train, X_val, y_train, y_val = split_data(padded_samples, labels)

y_train = [label - 1 for label in y_train]
y_val = [label - 1 for label in y_val]

print("Train labels(encoded):", np.unique(y_train))
print("Val labels(encoded):", np.unique(y_val))

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

batch_size = 16
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0
)

# === Model ===
model = Attention_over_timestep_LSTMClassifier(
    input_size=6,
    hidden_size=64,
    num_layers=2,
    dropout=0.4152860698857683,  # <-- Dropout integrated into model
    num_classes=4
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,  # 🔽 lowered from 0.009 for stability
    weight_decay=0.00031005513921773646
)

# === Training with gradient clipping ===
def train_with_clipping(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            
            # ⚠️ NaN check
            if torch.isnan(loss):
                print("❌ NaN loss encountered, stopping training.")
                return train_losses, val_losses, train_accs, val_accs
            
            loss.backward()

            # ✅ Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = epoch_loss / total
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_losses.append(val_loss / val_total)
        val_accs.append(100 * val_correct / val_total)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_losses[-1]:.4f} | Validation Accuracy: {val_accs[-1]:.2f}%")

    return train_losses, val_losses, train_accs, val_accs

# === Train ===
train_losses, val_losses, train_accs, val_accs = train_with_clipping(
    model, train_loader, val_loader, criterion, optimizer, epochs=1000
)

# === Learning Curve ===
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label='Train Accuracy')
plt.plot(epochs_range, val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.savefig("final_learning_curve_atten_overtslstm.png")
print("📊 Learning curve saved to final_learning_curve_atten_overtslstm.png")

# === Check for NaNs before saving ===
for name, param in model.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"⚠️ Warning: Parameter {name} has NaNs or Infs!")

# === Save model safely ===
model_cpu = model.to("cpu")
with open("lstm.pth", "wb") as f:
    torch.save(model_cpu.state_dict(), f)
model.to(device)
print("✅ Model saved to lstm.pth")
