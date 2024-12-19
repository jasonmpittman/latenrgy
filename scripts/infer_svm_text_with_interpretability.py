import os
import time
import numpy as np
import joblib
from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rai_interpretability import interpret_with_lime

# Paths and Directories
MODEL_PATH = "models/svm_text_model.pkl"
DATA_DIR = "data/inference_text"
RESULTS_DIR = "results/lime_svm_interpretability"
LIME_OUTPUT_DIR = os.path.join(RESULTS_DIR, "lime_svm_interpretability")

# Ensure results directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LIME_OUTPUT_DIR, exist_ok=True)

def load_inference_data():
    """ Load preprocessed text data for inference """
    print("Loading inference data...")
    X_data = np.load(os.path.join(DATA_DIR, "X_inference.npy"), allow_pickle=True)
    y_data = np.load(os.path.join(DATA_DIR, "y_inference.npy"), allow_pickle=True)

    # Convert bytes-like objects to strings if necessary
    if isinstance(X_data[0], bytes):
        X_data = np.array([x.decode('utf-8') for x in X_data])

    print(f"Sample after decoding: {X_data[0]}")
    return X_data, y_data


def load_model():
    """ Load the trained SVM model """
    print(f"Loading SVM model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    return model

def evaluate_model(model, X_data, y_data):
    """ Run inference, measure latency, and capture metrics, including interpretability """
    print("\n--- Running Inference, Evaluation, and Interpretability ---")
    
    # Energy tracker
    tracker = EmissionsTracker(allow_multiple_runs=True, output_file="emissions/emissions_svm_text_with_interpretability.csv")
    tracker.start()

    start_time = time.time()

    # Run inference
    y_pred = model.predict(X_data)
    
    # Generate interpretability outputs
    interpret_with_lime(model, X_data, LIME_OUTPUT_DIR, instance_indices=[0, 1, 2, 3, 4])

    end_time = time.time()
    
    # Stop energy tracker
    emissions = tracker.stop()

    # Calculate latency
    latency = (end_time - start_time) / len(X_data)
    
    # Performance Metrics
    accuracy = accuracy_score(y_data, y_pred)
    precision = precision_score(y_data, y_pred, average="binary")
    recall = recall_score(y_data, y_pred, average="binary")
    f1 = f1_score(y_data, y_pred, average="binary")

    # Log metrics
    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Latency (s/sample)": latency,
        "Energy Consumption (kWh)": emissions
    }

    print("\n--- Performance Metrics ---")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, "svm_text_with_interpretability_results.txt")
    with open(results_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Results saved to: {results_path}")
    
    return y_pred

def main():
    """ Main execution flow """
    # Load data and model
    X_data, y_data = load_inference_data()
    model = load_model()

    # Evaluate the model
    y_pred = evaluate_model(model, X_data, y_data)

if __name__ == "__main__":
    main()
