import os
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = "models/nn_tabular_model.pth"
X_INFERENCE_PATH = "data/inference_tabular/X_inference.npy"
Y_INFERENCE_PATH = "data/inference_tabular/y_inference.npy"
RESULTS_PATH = "results/nn_tabular_inference_results.json"
VISUALIZATIONS_DIR = "visualizations/tabular/nn"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Neural Network architecture (must match the training architecture)
class TabularNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TabularNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)  # Output for binary classification
        return x

def save_bar_chart(metrics):
    """
    Save a bar chart of the performance metrics.
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, metric_values, color="skyblue")
    plt.title("Neural Network Performance Metrics (Inference - Tabular)")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "performance_metrics_inference.png"))
    plt.close()

def main():
    # Load preprocessed data
    print("Loading preprocessed inference data...")
    X_inference = np.load(X_INFERENCE_PATH)
    y_true = np.load(Y_INFERENCE_PATH)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_inference, dtype=torch.float32).to(device)
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32).to(device)

    # Load the trained Neural Network model
    print("Loading trained Neural Network model...")
    input_size = X_inference.shape[1]
    num_classes = 2  # Set this to the number of classes used in training
    model = TabularNN(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Initialize energy tracker
    tracker = EmissionsTracker(output_file="emissions/emissions_tabular_nn_inference.csv", allow_multiple_runs=True)
    tracker.start()

    # Perform inference
    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()  # Model output
        if outputs.ndim == 2:  # Check for multilabel-indicator format
            y_pred = outputs.argmax(axis=1).cpu().numpy()  # Convert to binary
        else:
            y_pred = (outputs > 0.5).int().cpu().numpy()  # Binary threshold for probabilities
    latency = time.time() - start_time

    # Ensure y_true is in binary format
    if y_true.ndim == 2:  # Multilabel-indicator format
        y_true = y_true.argmax(axis=1)  # Convert to binary

    # Stop energy tracker
    emissions = tracker.stop()

    # Evaluate performance
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="binary"),
        "Recall": recall_score(y_true, y_pred, average="binary"),
        "F1 Score": f1_score(y_true, y_pred, average="binary"),
    }


    # Save confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    plt.title("Neural Network Confusion Matrix (Inference - Tabular)")
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "confusion_matrix_inference.png"))
    plt.close()

    # Save bar chart of metrics
    save_bar_chart(metrics)

    # Save results
    results = {
        "Latency (s)": latency,
        "Energy Consumption (kWh)": emissions,
        "Metrics": metrics,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        import json
        json.dump(results, f, indent=4)

    # Print results for debugging
    print(f"Latency: {latency:.4f} seconds")
    print(f"Energy Consumption: {emissions:.4f} kWh")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
