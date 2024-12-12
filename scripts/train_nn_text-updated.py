import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from utils import save_results

# Paths
X_TRAIN_PATH = "data/text/X_train.npy"
Y_TRAIN_PATH = "data/text/y_train.npy"
MODEL_PATH = "models/nn_text_model.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Neural Network model
class TextClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    X = np.load(X_TRAIN_PATH)
    y = np.load(Y_TRAIN_PATH)

    # Split into training and validation sets
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val_torch = torch.tensor(y_val, dtype=torch.long).to(device)

    # Model configuration
    input_size = X_train_torch.shape[1]
    num_classes = len(set(y))
    model = TextClassifier(input_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start energy tracker
    tracker = EmissionsTracker(allow_multiple_runs=True, output_file="emissions/emissions_nn_text.csv")
    tracker.start()

    # Train the model
    print("Training Neural Network...")
    start_time = time.time()
    epochs = 10
    batch_size = 32

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, X_train_torch.size(0), batch_size):
            X_batch = X_train_torch[i:i + batch_size]
            y_batch = y_train_torch[i:i + batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    training_time = time.time() - start_time

    # Stop energy tracker
    emissions = tracker.stop()

    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to: {os.path.abspath(MODEL_PATH)}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_torch)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()

    metrics = {
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(y_val, predictions, average="weighted"),
        "Recall": recall_score(y_val, predictions, average="weighted"),
        "F1 Score": f1_score(y_val, predictions, average="weighted"),
    }

    # Save results
    save_results("nn_text", metrics, training_time, emissions)

    # Print results for debugging
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Energy Consumption: {emissions:.2f} kWh")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
