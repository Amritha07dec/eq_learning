from train import (
    pad_samples,
    TimeSeriesDataset,
    LSTMClassifier,
    validate,
    plot_confusion_matrix,
    Conv1DLSTMClassifier,
    Attention_over_timestep_LSTMClassifier,
    Attention_over_features_LSTMClassifier,
    FrequencyAttentionClassifier
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data_preprocessing.delete import time_series_list, labels, filenames
import csv

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data preprocessing
padded_samples = pad_samples(time_series_list, target_dim=6)
labels = [label - 1 for label in labels]  # Shift labels to 0-based indexing

# Dataset and DataLoader
unseen_dataset = TimeSeriesDataset(padded_samples, labels)
unseen_loader = DataLoader(unseen_dataset, batch_size=16, shuffle=False)

# Load the model
model = Attention_over_timestep_LSTMClassifier(
    input_size=6,
    hidden_size=64,
    num_layers=2,
    dropout=0.4152860698857683,  # <-- Dropout integrated into model
    num_classes=4
)
model.load_state_dict(torch.load("/home/guest/Amritha/lstm.pth", map_location=device))
model.to(device)

# Evaluation
criterion = nn.CrossEntropyLoss()
val_loss, val_acc, preds, confs, targets, probs = validate(model, unseen_loader, criterion, return_all=True)

# Print predictions
for i in range(len(preds)):
    print(f"{filenames[i]}: True={targets[i]}, Pred={preds[i]}, Confidence={confs[i]:.4f}")
    print(f"  Probabilities: {np.round(probs[i], 4)}")

print(f"Unseen Data - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

# Plot confusion matrix
plot_confusion_matrix(model, unseen_loader, "conf_matrix_final_lstm.png")

# Save predictions to CSV
with open("final_predictions_atten_overts_lstm.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "True Label", "Predicted Label", "Confidence", "Probabilities"])
    for i in range(len(preds)):
        row = [
            filenames[i],
            targets[i],
            preds[i],
            round(confs[i], 4),
            np.round(probs[i], 4).tolist()
        ]
        writer.writerow(row)

print("Results saved to final_predictions_atten_overts_lstm.csv")
