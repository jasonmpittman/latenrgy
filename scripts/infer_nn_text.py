import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import json

# Paths
MODEL_PATH = "models/nn_text_model.pth"
X_INFERENCE_PATH = "data/inference_text/X_inference.npy"
Y_INFERENCE_PATH = "data/inference_text/y_inference.npy"
RESULTS_PATH = "results/nn_text_inference_results.json"
VISUALIZATIONS_DIR = "visualizations/text/nn"

# Ensure visualization directory exists
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Neural Network model (same architecture as used during training)
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

def save_bar_chart(metrics):
    """
    Save a bar chart of performance metrics.
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, metric_values, color="skyblue")
    plt.title("Neural Network Performance Metrics (Inference - Text)")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "performance_metrics_inference.png"))
    plt.close()

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load inference data
    print("Loading preprocessed inference data...")
    X_inference = np.load(X_INFERENCE_PATH)
    if os.path.exists(Y_INFERENCE_PATH):
        y_true = np.load(Y_INFERENCE_PATH)
    else:
        y_true = None

    # Convert inference data to PyTorch tensors
    X_inference_torch = torch.tensor(X_inference, dtype=torch.float32).to(device)

    # Load the trained Neural Network model
    print("Loading trained Neural Network model...")
    input_size = X_inference.shape[1]
    num_classes = len(set(y_true)) if y_true is not None else 2
    model = TextClassifier(input_size, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Initialize energy tracker
    tracker = EmissionsTracker(allow_multiple_runs=True, output_file="emissions/emissions_nn_text_inference.csv")
    tracker.start()

    # Perform inference
    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(X_inference_torch)
        _, y_pred = torch.max(outputs, 1)
    latency = time.time() - start_time

    # Stop energy tracker
    emissions = tracker.stop()

    # Evaluate performance metrics if labels are available
    metrics = {}
    if y_true is not None:
        y_pred_np = y_pred.cpu().numpy()
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred_np),
            "Precision": precision_score(y_true, y_pred_np, average="weighted"),
            "Recall": recall_score(y_true, y_pred_np, average="weighted"),
            "F1 Score": f1_score(y_true, y_pred_np, average="weighted"),
        }

        # Generate and save confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred_np, cmap="Blues")
        plt.title("Neural Network Confusion Matrix (Inference - Text)")
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, "confusion_matrix_inference.png"))
        plt.close()

        # Save bar chart of metrics
        save_bar_chart(metrics)

    # Save results to a JSON file
    results = {
        "Latency (s)": latency,
        "Energy Consumption (kWh)": emissions,
        "Metrics": metrics,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    # Print results for debugging
    print(f"Latency: {latency:.4f} seconds")
    print(f"Energy Consumption: {emissions:.4f} kWh")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
