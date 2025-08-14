import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random

# === Import ===
from train import pad_samples, split_data, TimeSeriesDataset, train
from data_preprocessing.delete import time_series_list, labels

# === Attention over Features Model ===
class AttentionOverFeaturesLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=1, num_classes=4, dropout=0.0, attn_mode="temporal"):
        super(AttentionOverFeaturesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn_layer = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.attn_mode = attn_mode

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden)

        if self.attn_mode == "temporal":
            attn_scores = self.attn_layer(lstm_out)               # (batch, time, hidden)
            attn_weights = F.softmax(attn_scores, dim=2)          # softmax over features
            context = torch.sum(attn_weights * lstm_out, dim=1)   # aggregate over time

        elif self.attn_mode == "global":
            pooled = torch.mean(lstm_out, dim=1)                  # (batch, hidden)
            attn_scores = self.attn_layer(pooled)                 # (batch, hidden)
            attn_weights = F.softmax(attn_scores, dim=1)          # softmax over features
            context = attn_weights * pooled                       # (batch, hidden)

        else:
            raise ValueError(f"Unknown attention mode: {self.attn_mode}")

        return self.fc(context)

# === Device Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Seeding ===
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

    # --- Hyperparameter Space ---
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    attn_mode = trial.suggest_categorical("attn_mode", ["temporal", "global"])

    # --- Dataset ---
    padded = pad_samples(time_series_list, target_dim=6)
    labels_encoded = [label - 1 for label in labels]
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
    model = AttentionOverFeaturesLSTM(
        input_size=6,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=4,
        attn_mode=attn_mode
    ).to(device)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Train + Eval ---
    try:
        _, _, _, val_accs = train(model, train_loader, val_loader, criterion, optimizer, epochs=30)
        return max(val_accs)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("⚠️ CUDA OOM - Skipping trial.")
            torch.cuda.empty_cache()
            return 0.0
        else:
            raise e

# === Run Optuna Study ===
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=123))
    study.optimize(objective, n_trials=30)

    print("\n✅ Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"🎯 Best Validation Accuracy: {study.best_value:.4f}")
