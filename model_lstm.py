from train import (
    pad_samples, split_data, TimeSeriesDataset,
    LSTMClassifier, train, Conv1DLSTMClassifier, set_seed
)
from data_preprocessing.delete import time_series_list, labels
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn

# === Seed setup ===
seed = 123
set_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Data preparation ===
padded_samples = pad_samples(time_series_list, target_dim=6)
X_train, X_val, y_train, y_val = split_data(padded_samples, labels)

# Shift labels to range [0, 3]
y_train = [label - 1 for label in y_train]
y_val = [label - 1 for label in y_val]

print("Train labels (encoded):", np.unique(y_train))
print("Val labels (encoded):", np.unique(y_val))

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

# ✅ Use best batch size from Optuna (16)
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True,
    worker_init_fn=seed_worker, generator=g, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False,
    worker_init_fn=seed_worker, generator=g, num_workers=0
)

# === Model with best Optuna hyperparameters ===
model = LSTMClassifier(
    input_size=6,
    hidden_size=128,
    num_layers=2,
    dropout=0.25974255962990467,  # dropout only applies if num_layers > 1
    num_classes=4
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0006897370850315688,
    weight_decay=3.0374886510756914e-06
)

# === Training ===
train_losses, val_losses, train_accs, val_accs = train(
    model, train_loader, val_loader, criterion, optimizer, epochs=200  # 200 is a good default
)

# === Learning curve plotting ===
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
plt.savefig("final_learning_curve_lstm.png")
print("📊 Learning curve saved to final_learning_curve_lstm.png")

# === Sanity check for NaNs/Infs ===
for name, param in model.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"⚠️ Warning: Parameter {name} has NaNs or Infs!")

# === Save model ===
model_cpu = model.to("cpu")
with open("lstm.pth", "wb") as f:
    torch.save(model_cpu.state_dict(), f)
model.to(device)
print("✅ Model saved to lstm.pth")
