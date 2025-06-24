from train import (
    pad_samples,
    TimeSeriesDataset,
    LSTMClassifier,
    validate,
    plot_confusion_matrix, Conv1DLSTMClassifier, AttentionLSTMClassifier
)
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data_preprocessing.delete import time_series_list, labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

padded_samples = pad_samples(time_series_list, target_dim=6)
labels = [label - 1 for label in labels]


unseen_dataset = TimeSeriesDataset(padded_samples, labels)
unseen_loader = DataLoader(unseen_dataset, batch_size=64, shuffle=False)


model = Conv1DLSTMClassifier()
model.load_state_dict(torch.load("/home/guest/Amritha/best_model.pth", map_location=device))
model.to(device)


criterion = nn.CrossEntropyLoss()
#val_loss, val_acc = validate(model, unseen_loader, criterion)
val_loss, val_acc, preds, confs, targets, probs = validate(model, unseen_loader, criterion, return_all=True)
for i in range(len(preds)):
    print(f"Sample {i}: True={targets[i]}, Pred={preds[i]}, Confidence={confs[i]:.4f}")
    print(f"  Probabilities: {np.round(probs[i], 4)}")

print(f"Unseen Data - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")


plot_confusion_matrix(model, unseen_loader, "conf_matrix_unseen.png")
