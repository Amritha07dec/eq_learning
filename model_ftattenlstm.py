# train_model.py
from train import (
    pad_samples, split_data, TimeSeriesDataset,
    LSTMClassifier, train, Conv1DLSTMClassifier, Attention_over_timestep_LSTMClassifier, Attention_over_features_LSTMClassifier, set_seed
)
from data_preprocessing.delete import time_series_list, labels
import numpy as np
import torch
import random

#labels = [label - 1 for label in labels]
seed=123
set_seed(seed)
# ✅ Ensures deterministic DataLoader behavior
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)  # Same seed as in set_seed()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Prepare data
padded_samples = pad_samples(time_series_list, target_dim=6)
X_train, X_val, y_train, y_val = split_data(padded_samples, labels)
print("Train labels:", np.unique(y_train))
print("Val labels:", np.unique(y_val))
#The labels range has to be [0,3]
y_train = [label - 1 for label in y_train]
y_val = [label - 1 for label in y_val]

# After splitting
#X_train, X_val, y_train, y_val = split_data(padded_samples, labels)

# ✅ Add this to inspect unique label values

print("Train labels(encoded):", np.unique(y_train))
print("Val labels(encoded):", np.unique(y_val))

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0
)


# Initialize model, criterion, optimizer
model = Attention_over_features_LSTMClassifier(
    input_size=6,
    hidden_size=64,
    num_layers=1,
    num_classes=4,
    attn_mode="temporal"  # or "global"
)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train model
#train(model, train_loader, val_loader, criterion, optimizer, epochs=40)
train_losses, val_losses, train_accs, val_accs = train(
    model, train_loader, val_loader, criterion, optimizer, epochs=1000
)
###Learning curve plots###
import matplotlib.pyplot as plt

epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Accuracy plot
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

#Add a quick check for NANs/INF before saving
for name, param in model.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"⚠️ Warning: Parameter {name} has NaNs or Infs!")
# Save model
# torch.save(model.state_dict(), "today_Conv1DLSTM_model.pth")
# print("✅ Model saved to today_Conv1DLSTM_model.pth")

# Move model to CPU before saving (avoids CUDA memory pointer issues)
model_cpu = model.to("cpu")

# Save using context manager
with open("lstm.pth", "wb") as f:
    torch.save(model_cpu.state_dict(), f)

# Move it back to original device (optional)
model.to(device)

###torch.save(model.state_dict(), "today_lstm_model.pth")
print("✅ Model saved to lstm.pth")