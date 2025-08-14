from train import (
    pad_samples, split_data, TimeSeriesDataset,
    LSTMClassifier, train, Conv1DLSTMClassifier, set_seed
)
from data_preprocessing.delete import time_series_list, labels
import numpy as np
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ----------------------------
# ✅ Best Hyperparameters from Optuna for idk when lol
# ----------------------------
# best_params = {
#     'conv_channels': 32,
#     'kernel_size': 3,
#     'lstm_hidden': 64,
#     'dropout': 0.008231293935378081,
#     'lr': 0.007849862578644864,
#     'weight_decay': 1.1596026993791032e-06,
#     'batch_size': 16
# }
best_params = {
    'conv_channels': 64,
    'kernel_size': 5,
    'lstm_hidden': 64,
    'dropout': 0.13171037835236465,
    'lr': 0.0020161453334313975,
    'weight_decay': 2.2207215073249646e-06,
    'batch_size': 16
}


# ----------------------------
# ✅ Seeding
# ----------------------------
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

# ----------------------------
# ✅ Data Preparation
# ----------------------------
padded_samples = pad_samples(time_series_list, target_dim=6)
X_train, X_val, y_train, y_val = split_data(padded_samples, labels)

y_train = [label - 1 for label in y_train]
y_val = [label - 1 for label in y_val]

print("Train labels(encoded):", np.unique(y_train))
print("Val labels(encoded):", np.unique(y_val))

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=best_params['batch_size'],
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=best_params['batch_size'],
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0
)

# ----------------------------
# ✅ Model, Criterion, Optimizer
# ----------------------------
model = Conv1DLSTMClassifier(
    input_size=6,
    conv_channels=best_params['conv_channels'],
    kernel_size=best_params['kernel_size'],
    lstm_hidden=best_params['lstm_hidden'],
    dropout=best_params['dropout'],
    num_classes=4
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=best_params['lr'],
    weight_decay=best_params['weight_decay']
)

# ----------------------------
# ✅ Training with Early Stopping
# ----------------------------
train_losses, val_losses, train_accs, val_accs = train(
    model, train_loader, val_loader, criterion, optimizer, epochs=1000
)

# ----------------------------
# ✅ Plotting Learning Curves
# ----------------------------
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label='Train Accuracy')
plt.plot(epochs_range, val_accs, label='Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.savefig("final_learning_curve_convlstm.png")
print("📊 Learning curve saved to final_learning_curve_convlstm.png")

# ----------------------------
# ✅ Sanity check before saving
# ----------------------------
for name, param in model.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"⚠️ Warning: Parameter {name} has NaNs or Infs!")

# ----------------------------
# ✅ Save final model
# ----------------------------
model_cpu = model.to("cpu")
with open("lstm.pth", "wb") as f:
    torch.save(model_cpu.state_dict(), f)
print("✅ Model saved to lstm.pth")

model.to(device)  # optional: return to GPU
