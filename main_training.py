import os
import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import json

# Paths
SCRIPT_DIR = "scripts"
RESULTS_DIR = "results"
VISUALIZATIONS_DIR = "visualizations"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Datasets and models
datasets = ["image", "text", "tabular"]
models = ["svm", "knn", "rf", "nn"]

def preprocess_data(dataset):
    """
    Run the preprocessing script for a given dataset.
    """
    script_path = os.path.join(SCRIPT_DIR, f"preprocess_{dataset}.py")
    print(f"Preprocessing {dataset} data...")
    subprocess.run(["python3", script_path], check=True)

def train_model(dataset, model):
    """
    Train a specific model on a specific dataset.
    """
    script_path = os.path.join(SCRIPT_DIR, f"train_{model}_{dataset}.py")
    print(f"Training {model.upper()} on {dataset} data...")
    subprocess.run(["python3", script_path], check=True)

def generate_visualizations(dataset, model):
    """
    Generate visualizations for the trained model on a specific dataset.
    """
    results_path = os.path.join(RESULTS_DIR, f"{model}_{dataset}_results.json")
    vis_dir = os.path.join(VISUALIZATIONS_DIR, dataset, model)
    os.makedirs(vis_dir, exist_ok=True)

    # Load results
    with open(results_path, "r") as f:
        results = json.load(f)

    # Example: Visualize accuracy, precision, recall, F1
    metrics = results["Metrics"]
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # Bar chart for metrics
    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, metric_values, color="skyblue")
    plt.title(f"{model.upper()} Metrics for {dataset}")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(vis_dir, "metrics.png"))
    plt.close()

    # Example: Add confusion matrix visualization (if applicable)
    if "Confusion Matrix" in results:
        cm = results["Confusion Matrix"]
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
        plt.title(f"{model.upper()} Confusion Matrix for {dataset}")
        plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"))
        plt.close()

# Main workflow
for dataset in datasets:
    preprocess_data(dataset)
    for model in models:
        train_model(dataset, model)
        generate_visualizations(dataset, model)

print("All training and visualizations completed.")
