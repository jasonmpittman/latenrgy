import os
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from codecarbon import EmissionsTracker
from rai_explainability import explain_with_precomputed_shap

# Directories
DATA_DIR = "data/inference_text"
MODEL_DIR = "models"
SHAP_DIR = "data/inference_text/shap_values"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """Load preprocessed inference data."""
    X_path = os.path.join(DATA_DIR, "X_inference.npy")
    y_path = os.path.join(DATA_DIR, "y_inference.npy")
    print("Loading preprocessed inference data...")
    X = np.load(X_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    return X, y

def load_model():
    """Load the trained k-NN model."""
    model_path = os.path.join(MODEL_DIR, "knn_text_model.pkl")
    print(f"Loading k-NN model from {model_path}...")
    import joblib
    return joblib.load(model_path)

def run_inference(model, X):
    """Run inference and return predictions."""
    print("Running inference...")
    return model.predict(X)

def evaluate_performance(y_true, y_pred):
    """Evaluate model performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    return accuracy, f1, precision, recall

def main():
    # Start CodeCarbon tracker
    tracker = EmissionsTracker(allow_multiple_runs=True, output_file="emissions/emissions_knn_text_inference_rai.csv")
    tracker.start()

    # Load data and model
    X_inference, y_inference = load_data()
    knn_model = load_model()

    # Start latency timing
    start_time = time.time()

    # Run inference
    y_pred = run_inference(knn_model, X_inference)

    # Evaluate performance
    accuracy, f1, precision, recall = evaluate_performance(y_inference, y_pred)

    # Run SHAP explainability
    print("\n--- Running SHAP Explainability ---")
    explain_with_precomputed_shap(knn_model, X_inference, SHAP_DIR, model_type="k-NN", dataset="text")

    # Stop latency timing and energy tracker
    end_time = time.time()
    emissions = tracker.stop()

    # Handle None case for emissions
    emissions_str = f"{emissions:.4f} kWh" if emissions is not None else "Energy data unavailable"

    # Print and save results
    print("\n--- Performance Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Total Latency (including explainability): {end_time - start_time:.4f} seconds")
    print(f"Energy Consumption: {emissions_str}")

    results_path = os.path.join(RESULTS_DIR, "knn_text_inference_with_rai_results.txt")
    with open(results_path, "w") as f:
        f.write("--- k-NN Text Inference with RAI Results ---\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Total Latency: {end_time - start_time:.4f} seconds\n")
        f.write(f"Energy Consumption: {emissions_str}\n")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
