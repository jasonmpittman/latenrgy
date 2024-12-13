import os
import time
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import json

# Paths
MODEL_PATH = "models/rf_text_model.pkl"
X_INFERENCE_PATH = "data/inference_text/X_inference.npy"
Y_INFERENCE_PATH = "data/inference_text/y_inference.npy"
RESULTS_PATH = "results/rf_text_inference_results.json"
VISUALIZATIONS_DIR = "visualizations/text/rf"

# Ensure visualization directory exists
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

def save_bar_chart(metrics):
    """
    Save a bar chart of performance metrics.
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, metric_values, color="skyblue")
    plt.title("Random Forest Performance Metrics (Inference - Text)")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "performance_metrics_inference.png"))
    plt.close()

def main():
    # Load inference data
    print("Loading preprocessed inference data...")
    X_inference = np.load(X_INFERENCE_PATH)
    if os.path.exists(Y_INFERENCE_PATH):
        y_true = np.load(Y_INFERENCE_PATH)
    else:
        y_true = None

    # Load the trained Random Forest model
    print("Loading trained Random Forest model...")
    rf = joblib.load(MODEL_PATH)

    # Initialize energy tracker
    tracker = EmissionsTracker(allow_multiple_runs=True, output_file="emissions/emissions_rf_text_inference.csv")
    tracker.start()

    # Perform inference
    print("Running inference...")
    start_time = time.time()
    y_pred = rf.predict(X_inference)
    latency = time.time() - start_time

    # Stop energy tracker
    emissions = tracker.stop()

    # Evaluate performance metrics if labels are available
    metrics = {}
    if y_true is not None:
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted"),
            "Recall": recall_score(y_true, y_pred, average="weighted"),
            "F1 Score": f1_score(y_true, y_pred, average="weighted"),
        }

        # Generate and save confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
        plt.title("Random Forest Confusion Matrix (Inference - Text)")
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
