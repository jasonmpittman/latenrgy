import os
import time
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt

# Paths
DATA_DIR = "data/inference_images"
MODEL_PATH = "models/svm_image_model.pkl"
RESULTS_PATH = "results/svm_image_inference_results.json"
VISUALIZATIONS_DIR = "visualizations/image/svm"

# Load preprocessed data
def load_data():
    images_path = os.path.join(DATA_DIR, "images.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")
    images = np.load(images_path)
    labels = np.load(labels_path)
    return images, labels

def save_bar_chart(metrics):
    """
    Save a bar chart of the performance metrics.
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, metric_values, color="skyblue")
    plt.title("SVM Performance Metrics (Inference - Images)")
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
    print("Loading trained SVM model...")
    model = joblib.load(MODEL_PATH)

    # Initialize energy tracker
    tracker = EmissionsTracker(output_file="emissions/emissions_image_inference.csv", allow_multiple_runs=True)
    tracker.start()

    # Measure inference latency
    print("Running inference...")
    start_time = time.time()
    y_pred = model.predict(X.reshape(X.shape[0], -1))  # Flatten images for SVM input
    latency = time.time() - start_time

    # Stop energy tracker
    emissions = tracker.stop()

    # Evaluate performance
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
    }

    # Save confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    plt.title("SVM Confusion Matrix (Inference - Images)")
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