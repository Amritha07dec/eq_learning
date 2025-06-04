# train_model.py
from train import (
    pad_samples, split_data, TimeSeriesDataset,
    LSTMClassifier, train, Conv1DLSTMClassifier
)
from data_preprocessing.delete import time_series_list, labels
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize model, criterion, optimizer
model = LSTMClassifier()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
#train(model, train_loader, val_loader, criterion, optimizer, epochs=40)
train_losses, val_losses, train_accs, val_accs = train(
    model, train_loader, val_loader, criterion, optimizer, epochs=10
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
plt.savefig("learning_curve.png")
print("📊 Learning curve saved to learning_curve.png")

#Add a quick check for NANs/INF before saving
for name, param in model.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"⚠️ Warning: Parameter {name} has NaNs or Infs!")

# Save model
##########Use the following code instead ofthe triple hashed out code


# Move model to CPU before saving (avoids CUDA memory pointer issues)
model_cpu = model.to("cpu")

# Save using context manager
with open("1_layer_lstm_model_1312.pth", "wb") as f:
    torch.save(model_cpu.state_dict(), f)

# Move it back to original device (optional)
model.to(device)

###torch.save(model.state_dict(), "today_lstm_model.pth")
print("✅ Model saved to 1_layer_lstm_model_1312.pth")