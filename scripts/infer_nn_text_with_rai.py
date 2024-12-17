import os
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from codecarbon import EmissionsTracker
from rai_explainability import explain_with_precomputed_shap

# Directories
DATA_DIR = "data/inference_text"
MODEL_DIR = "models"
SHAP_DIR = "data/inference_text/shap_values"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Neural Network Definition
class TextNN(torch.nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(TextNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)  # Output layer
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

def load_data():
    """Load preprocessed inference data."""
    X_path = os.path.join(DATA_DIR, "X_inference.npy")
    y_path = os.path.join(DATA_DIR, "y_inference.npy")
    print("Loading preprocessed inference data...")
    X = np.load(X_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    return X, y

def load_model(input_size):
    """Load the trained Neural Network model."""
    model_path = os.path.join(MODEL_DIR, "nn_text_model.pth")
    print(f"Loading Neural Network model from {model_path}...")
    model = TextNN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def run_inference(model, X):
    """Run inference and return predictions."""
    print("Running inference...")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    return predictions

def evaluate_performance(y_true, y_pred):
    """Evaluate model performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    return accuracy, f1, precision, recall

def main():
    # Start emissions tracker
    tracker = EmissionsTracker(allow_multiple_runs=True, output_file="emissions/emissions_nn_text_inference_rai.csv")
    tracker.start()

    # Load data and model
    X_inference, y_inference = load_data()
    input_size = X_inference.shape[1]
    nn_model = load_model(input_size)

    # Measure total latency including SHAP explainability
    start_time = time.time()
    
    # Run inference
    y_pred = run_inference(nn_model, X_inference)

    # Evaluate performance
    accuracy, f1, precision, recall = evaluate_performance(y_inference, y_pred)

    # Run SHAP explainability
    print("\n--- Running SHAP Explainability ---")
    explain_with_precomputed_shap(nn_model, X_inference, SHAP_DIR, model_type="NeuralNetwork", dataset="text")

    # Measure total latency and stop tracker
    end_time = time.time()
    emissions = tracker.stop()

    # Print results
    print("\n--- Performance Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Total Latency (including explainability): {end_time - start_time:.4f} seconds")
    print(f"Energy Consumption: {emissions:.4f} kWh")

    # Save results to file
    results_path = os.path.join(RESULTS_DIR, "nn_text_inference_with_rai_results.txt")
    with open(results_path, "w") as f:
        f.write("--- Neural Network Text Inference with RAI Results ---\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Total Latency: {end_time - start_time:.4f} seconds\n")
        f.write(f"Energy Consumption: {emissions:.4f} kWh\n")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
