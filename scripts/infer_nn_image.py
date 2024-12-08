import os
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt

# Paths
DATA_DIR = "data/inference_image"
MODEL_PATH = "models/nn_image_model.pth"
RESULTS_PATH = "results/nn_image_inference_results.json"
VISUALIZATIONS_DIR = "visualizations/image/nn"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load preprocessed data
def load_data():
    images_path = os.path.join(DATA_DIR, "images.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")
    images = np.load(images_path)
    labels = np.load(labels_path)
    return images, labels

# Define the Neural Network architecture (must match training architecture)
class ImageNN(nn.Module):
    def __init__(self):
        super(ImageNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 31 * 31, 128)  # Matches the flattened input size
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.contiguous().view(x.size(0), -1)  # Flattening step
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Corrected: Pass x to Sigmoid
        return x

def save_bar_chart(metrics):
    """
    Save a bar chart of the performance metrics.
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, metric_values, color="skyblue")
    plt.title("Neural Network Performance Metrics (Inference - Images)")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "performance_metrics_inference.png"))
    plt.close()

def main():
    # Load data
    print("Loading preprocessed data...")
    X, y_true = load_data()

    # Load trained model
    print("Loading trained Neural Network model...")
    model = ImageNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)  # Convert to channels-first
    y_true_tensor = torch.tensor(y_true, dtype=torch.long).to(device)

    # Initialize energy tracker
    tracker = EmissionsTracker(output_file="emissions/emissions_image_nn_inference.csv", allow_multiple_runs=True)
    tracker.start()

    # Measure inference latency
    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(X_tensor)
        y_pred = (outputs > 0.5).int().squeeze(1)  # Threshold sigmoid output for binary classification
    latency = time.time() - start_time

    # Stop energy tracker
    emissions = tracker.stop()

    # Convert predictions to NumPy for metrics
    y_pred = y_pred.cpu().numpy()
    y_true = y_true_tensor.cpu().numpy()

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
    plt.title("Neural Network Confusion Matrix (Inference - Images)")
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
