import os
import time
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from codecarbon import EmissionsTracker
from rai_interpretability import interpret_with_lime

# Directories
DATA_DIR = "data/inference_text"
MODELS_DIR = "models"
RESULTS_DIR = "results"
EMISSIONS_DIR = "emissions"
LIME_OUTPUT_DIR = os.path.join(RESULTS_DIR, "lime_knn_interpretability")

# Ensure the output directories exist
os.makedirs(LIME_OUTPUT_DIR, exist_ok=True)
os.makedirs(EMISSIONS_DIR, exist_ok=True)

def load_inference_data():
    """ Load preprocessed inference data """
    print(f"Loading inference data from: {DATA_DIR}")
    X_data = np.load(os.path.join(DATA_DIR, "X_inference.npy"), allow_pickle=True)
    y_data = np.load(os.path.join(DATA_DIR, "y_inference.npy"), allow_pickle=True)
    return X_data, y_data

def load_model():
    """ Load the trained k-NN model """
    print(f"Loading k-NN model from: {MODELS_DIR}")
    model = joblib.load(os.path.join(MODELS_DIR, "knn_text_model.pkl"))
    return model

def evaluate_model(model, X_data, y_data):
    """ Evaluate model, capture latency and energy, and generate interpretability results """
    print("\n--- Running Inference, Evaluation, and Interpretability ---")
    
    # Energy tracker
    tracker = EmissionsTracker(allow_multiple_runs=True, output_file=os.path.join(EMISSIONS_DIR, "emissions_knn_text_with_interpretability.csv"))
    tracker.start()

    # Start timing
    start_time = time.time()

    # Run inference
    y_pred = model.predict(X_data)

    # Generate interpretability outputs
    interpret_with_lime(model, X_data, LIME_OUTPUT_DIR, instance_indices=[0, 1, 2, 3, 4])

    # End timing
    end_time = time.time()
    
    # Stop energy tracker
    emissions = tracker.stop()

    # Calculate latency
    latency = (end_time - start_time) / len(X_data)

    # Performance metrics
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
    results_path = os.path.join(RESULTS_DIR, "knn_text_with_interpretability_results.txt")
    with open(results_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Results saved to: {results_path}")
    
    return y_pred

def main():
    """ Main script execution """
    # Load data
    X_data, y_data = load_inference_data()

    # Load model
    knn_model = load_model()

    # Evaluate model with interpretability
    evaluate_model(knn_model, X_data, y_data)

if __name__ == "__main__":
    main()
