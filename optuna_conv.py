import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random

# === Import your modules ===
from train import Conv1DLSTMClassifier, train, pad_samples, split_data, TimeSeriesDataset
from data_preprocessing.delete import time_series_list, labels

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Seeding Functions ===
def set_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# === Objective Function ===
def objective(trial):
    torch.cuda.empty_cache()
    set_seed(123)

    # --- Hyperparameter Search Space ---
    conv_channels = trial.suggest_categorical("conv_channels", [16, 32, 64])  # changed name here
    kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)  # must be odd for padding
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    # Removed num_classes from search space — fixed at 4
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # --- Dataset Preparation ---
    padded = pad_samples(time_series_list, target_dim=6)
    labels_encoded = [label - 1 for label in labels]  # Assuming labels ∈ {1,2,3,4}
    X_train, X_val, y_train, y_val = split_data(padded, labels_encoded)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    g = torch.Generator()
    g.manual_seed(123)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              worker_init_fn=seed_worker, generator=g, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            worker_init_fn=seed_worker, generator=g, num_workers=0)

    # --- Model ---
    model = Conv1DLSTMClassifier(
        input_size=6,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        lstm_hidden=lstm_hidden,
        dropout=dropout,
        num_classes=4  # fixed
    ).to(device)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Train with OOM handling ---
    try:
        _, _, _, val_accs = train(model, train_loader, val_loader, criterion, optimizer, epochs=30)
        return max(val_accs)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("⚠️ CUDA out of memory, skipping trial.")
            torch.cuda.empty_cache()
            return 0.0
        else:
            raise e


# === Run the Optuna Study ===
if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=123))

    study.optimize(objective, n_trials=30)

    print("\n✅ Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"🎯 Best Validation Accuracy: {study.best_value:.4f}")
