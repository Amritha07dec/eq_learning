from train import pad_samples
from train import split_data
from train import TimeSeriesDataset
from train import DataLoader
from train import LSTMClassifier
from train import train
from train import plot_confusion_matrix
from data import time_series_list
from data import labels

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



#Pad the samples before feeding into the neural network
padded_samples = pad_samples(time_series_list, target_dim=6)  # adjust if your target feature size is different

# Step 9: Split into training and validation sets
X_train, X_val, y_train, y_val = split_data(padded_samples, labels, test_size=0.2)

# create dataset and datasloader
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Initialize your model loss and optimizer
model = LSTMClassifier(input_size=6, hidden_size=64, num_layers=1, num_classes=5)
model.to(device)  # 👈 move model to GPU


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, dataloader, val_loader, criterion, optimizer, epochs=2)

# Plot confusion matrix on validation set
plot_confusion_matrix(model, val_loader)

