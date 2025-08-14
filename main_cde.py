import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import datetime 

# Import the NeuralCDE model from your model_neural_cde.py file
from model_neuralcde import NeuralCDE

# Import the data loading utility
from dataloader import create_dataloaders 

# --- Configuration and Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the directories for training and unseen test data
TRAIN_DATA_DIRECTORY = 'Training_data.pkl' # Your training data folder
TEST_DATA_DIRECTORY = 'Validation_data.pkl'   # Your unseen test data folder

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

CONFUSION_MATRIX_OUTPUT_PATH = f'neural_cde_confusion_matrix_unseen_eval_{timestamp}.png'
MODEL_SAVE_PATH = f'neural_cde_model_trained_{timestamp}.pth' # Saved model will have this name

# --- Training Function ---
def train_model(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (times, y_values, labels) in enumerate(train_loader):
        times = times.to(device)
        y_values = y_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(times, y_values)
        loss = criterion(predictions, labels)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        
        _, predicted_labels = torch.max(predictions.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted_labels == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# --- Evaluation Function ---
def evaluate_model(model, data_loader, criterion, device, phase="Test"):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for times, y_values, labels in data_loader:
            times = times.to(device)
            y_values = y_values.to(device)
            labels = labels.to(device)
            predictions = model(times, y_values)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            _, predicted_labels = torch.max(predictions.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    print(f"{phase} Loss: {avg_loss:.4f} | {phase} Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# --- Confusion Matrix Plotting Function ---
def plot_confusion_matrix(model, data_loader, label_to_degree_map, matrix_file_name, device):
    print(f"\nPlotting confusion matrix and saving to {matrix_file_name}...")
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    class_names = [str(label_to_degree_map[i]) for i in sorted(label_to_degree_map.keys())]
    with torch.no_grad():
        for times, y_values, labels in data_loader:
            times = times.to(device)
            y_values = y_values.to(device)
            labels = labels.to(device)
            outputs = model(times, y_values)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Degree')
    plt.ylabel('Actual Degree')
    plt.title('Confusion Matrix for Neural CDE Predictions (Unseen Data)')
    plt.tight_layout()
    plt.savefig(matrix_file_name)
    print(f"Confusion matrix saved as {matrix_file_name}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Hyperparameters ---
    batch_size = 16
    learning_rate =1e-3
    num_epochs = 50
    hidden_channels = 128 # Set your desired hidden_channels for the NEW model training

    print(f"Using device: {device}")
    print("--- Loading Data ---")
    try:
        train_loader, test_loader, input_channels, num_classes, label_to_degree_map = create_dataloaders(
            TRAIN_DATA_DIRECTORY, TEST_DATA_DIRECTORY, batch_size=batch_size
        )
    except Exception as e:
        print(f"Error: Failed to load data. Please check your data directories and pickle files.")
        print(f"Details: {e}")
        exit()

    print(f"Determined Input Channels (dimensions of ODE state): {input_channels}")
    print(f"Determined Number of Output Classes (unique degrees): {num_classes}")
    print(f"Label to Degree Map: {label_to_degree_map}")

    print("\n--- Initializing Model ---")
    model = NeuralCDE(input_channels, hidden_channels, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # --- ADDED: Learning Rate Scheduler ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)#, verbose=True


    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        
        # Evaluate on test set after each epoch (or periodically) for scheduler and tracking
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, phase="Test (Unseen Data)")
        
        # --- ADDED: Scheduler Step ---
        # The scheduler steps based on the test_loss (validation loss)
        scheduler.step(test_loss)


    print("\n--- Training Complete! ---")

    print(f"\nSaving trained model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved successfully as {MODEL_SAVE_PATH}.")

    print("\n--- Final Evaluation on Unseen Test Set ---")
    final_test_loss, final_test_acc = evaluate_model(model, test_loader, criterion, device, phase="Final Test (Unseen Data)")
    print(f"Final Test Accuracy on Unseen Data: {final_test_acc:.4f}")

    plot_confusion_matrix(model, test_loader, label_to_degree_map, CONFUSION_MATRIX_OUTPUT_PATH, device)

    print("\n--- Example Prediction on a single unseen test sample ---")
    if len(test_loader.dataset) > 0:
        sample_idx = 0
        sample_times, sample_y_values, sample_true_label = test_loader.dataset[sample_idx]
        
        sample_times_batch = sample_times.unsqueeze(0).to(device)
        sample_y_values_batch = sample_y_values.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output_logits = model(sample_times_batch, sample_y_values_batch)
            _, predicted_label = torch.max(output_logits.data, 1)

        predicted_degree = label_to_degree_map[predicted_label.item()]
        true_degree = label_to_degree_map[sample_true_label.item()]

        print(f"Sample from Unseen Test Set (Index {sample_idx}):")
        print(f"  True Degree (Label {sample_true_label.item()}): {true_degree}")
        print(f"  Predicted Degree (Label {predicted_label.item()}): {predicted_degree}")
    else:
        print("No samples in the unseen test set to make a prediction.")