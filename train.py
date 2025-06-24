# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Padding
def pad_samples(raw_samples, target_dim=6):
    padded = []
    for ts in raw_samples:
        pad_width = target_dim - ts.shape[1]
        if pad_width > 0:
            ts = np.pad(ts, ((0, 0), (0, pad_width)), mode='constant')
        padded.append(ts)
    return np.array(padded)

# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=1, num_classes=4):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out
    
class Conv1DLSTMClassifier(nn.Module):
    def __init__(self, input_size=6, conv_channels=32, lstm_hidden=64, num_classes=4):
        super(Conv1DLSTMClassifier, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)
    def forward(self, x):
        # x: (batch, time, features) => transpose for Conv1D
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.relu(self.conv1d(x))  # Conv1D over time
        x = x.transpose(1, 2)  # (batch, time, conv_channels)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTMClassifier(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=1, num_classes=4):
        super(AttentionLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Attention layer (1-layer MLP that scores each hidden state)
        self.attn_layer = nn.Linear(hidden_size, 1)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, time, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, time, hidden_size)

        # Compute attention scores
        attn_scores = self.attn_layer(lstm_out)  # (batch, time, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, time, 1)

        # Compute context vector as weighted sum of hidden states
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)

        out = self.fc(context)  # (batch, num_classes)
        return out



# Train/Test split
def split_data(samples, labels, test_size=0.2):
    return train_test_split(samples, labels, test_size=test_size, random_state=42)

# Training
"""
def train(model, dataloader, val_loader, criterion, optimizer, epochs=10):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")
"""
####saves the best model
"""
def train(model, dataloader, val_loader, criterion, optimizer, epochs=10):
    model.to(device)
    best_val_acc = 0.0  # Track best validation accuracy

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Best model saved at Epoch {epoch+1} with Val Accuracy: {val_acc:.2f}%")
"""
def train(model, dataloader, val_loader, criterion, optimizer, epochs=10):
    model.to(device)
    best_val_acc = 0.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        train_acc = 100 * correct / total
        train_loss = total_loss / len(dataloader)

        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Best model saved at Epoch {epoch+1} with Val Accuracy: {val_acc:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies

# Validation
"""
def validate(model, val_loader, criterion):
    model.eval()
    model.to(device)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy
"""

def validate(model, val_loader, criterion, return_all=False):
    model.eval()
    model.to(device)
    total_loss = 0.0
    correct = 0
    total = 0

    all_confidences = []
    all_predictions = []
    all_targets = []
    all_probs = []  # Store full softmax distributions

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, 1)

            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            all_confidences.extend(confidences.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # Store full prob vector

    accuracy = 100 * correct / total
    avg_conf = np.mean(all_confidences)
    print(f"🔎 Average Confidence: {avg_conf:.4f}")

    # Print top 5 low-confidence predictions
    low_conf_indices = np.argsort(all_confidences)[:5]
    for i in low_conf_indices:
        print(f"⚠️ Low confidence: Pred={all_predictions[i]}, True={all_targets[i]}, Confidence={all_confidences[i]:.4f}")
        print(f"   Class Probabilities: {np.round(all_probs[i], 4)}")

    if return_all:
        return total_loss / len(val_loader), accuracy, all_predictions, all_confidences, all_targets, all_probs

    return total_loss / len(val_loader), accuracy



# Confusion Matrix
def plot_confusion_matrix(model, dataloader, matrix_file_name):
    print("Plotting confusion matrix...")
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())  
            all_labels.extend(targets.cpu().numpy())   


    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['class 1','Class 2', 'Class 3', 'Class 4'],
                yticklabels=['class 1','Class 2', 'Class 3', 'Class 4'])  #
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(matrix_file_name)
    print(f"Confusion matrix saved as {matrix_file_name}")
    plt.show() 