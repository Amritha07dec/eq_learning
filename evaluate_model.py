# evaluate_model.py
from train import LSTMClassifier,Conv1DLSTMClassifier, pad_samples, split_data, TimeSeriesDataset, plot_confusion_matrix
from data_preprocessing.delete import time_series_list, labels
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare validation data
padded_samples = pad_samples(time_series_list, target_dim=6)
_, X_val, _, y_val = split_data(padded_samples, labels)

val_dataset = TimeSeriesDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load model
model = Conv1DLSTMClassifier()
#model.load_state_dict(torch.load("Conv1DLSTM_model.pth", map_location=device))
model.to(device)

# Plot confusion matrix
plot_confusion_matrix(model, val_loader, matrix_file_name="conv1dlstm.png")
