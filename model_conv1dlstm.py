# train_model.py
from train import (
    pad_samples, split_data, TimeSeriesDataset,
    LSTMClassifier, train, Conv1DLSTMClassifier
)
from data_preprocessing.delete import time_series_list, labels
import numpy as np

#labels = [label - 1 for label in labels]

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, criterion, optimizer
model = Conv1DLSTMClassifier()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
train(model, train_loader, val_loader, criterion, optimizer, epochs=40)

# Save model
torch.save(model.state_dict(), "Conv1DLSTM_model.pth")
print("✅ Model saved to Conv1DLSTM_model.pth")
