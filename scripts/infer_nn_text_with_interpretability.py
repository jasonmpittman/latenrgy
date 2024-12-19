import os
import time
import joblib
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from codecarbon import EmissionsTracker
from rai_interpretability import interpret_with_lime

# Directories
DATA_DIR = "data/inference_text"
MODELS_DIR = "models"
RESULTS_DIR = "results"
EMISSIONS_DIR = "emissions"
LIME_OUTPUT_DIR = os.path.join(RESULTS_DIR, "lime_nn_interpretability")

# Ensure the output directories exist
os.makedirs(LIME_OUTPUT_DIR, exist_ok=True)
os.makedirs(EMISSIONS_DIR, exist_ok=True)

class TextNN(torch.nn.Module):
    """ Neural Network Architecture for Text Classification """
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

def load_inference_data():
    """ Load preprocessed inference data """
    print(f"Loading inference data from: {DATA_DIR}")
    X_data = np.load(os.path.join(DATA_DIR, "X_inference.npy"), allow_pickle=True)
    y_data = np.load(os.path.join(DATA_DIR, "y_inference.npy"), allow_pickle=True)
    return X_data, y_data

def load_model(input_size):
    """ Load the trained Neural Network model """
    print(f"Loading Neural Network model from: {MODELS_DIR}")
    model = TextNN(input_size=input_size)
    model_path = os.path.join(MODELS_DIR, "nn_text_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_vectorizer():
    """ Load the vectorizer directly using joblib """
    vectorizer_path = os.path.join(MODELS_DIR, "text_vectorizer.pkl")
    print(f"Loading vectorizer from: {vectorizer_path}")
    return joblib.load(vectorizer_path)

def evaluate_model(model, X_data, y_data):
    """ Evaluate model, capture latency and energy, and generate interpretability results """
    print("\n--- Running Inference, Evaluation, and Interpretability ---")

    # Convert data to tensors
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    
    # Energy tracker
    tracker = EmissionsTracker(allow_multiple_runs=True, output_file=os.path.join(EMISSIONS_DIR, "emissions_nn_text_with_interpretability.csv"))
    tracker.start()

    # Start timing
    start_time = time.time()

    # Run inference
    with torch.no_grad():
        y_pred = model(X_tensor).argmax(dim=1).numpy()

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
    results_path = os.path.join(RESULTS_DIR, "nn_text_with_interpretability_results.txt")
    with open(results_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Results saved to: {results_path}")
    
    return y_pred

def main():
    """ Main script execution """
    # Load data
    X_data, y_data = load_inference_data()

    # Load vectorizer to determine input size
    vectorizer = load_vectorizer()
    input_size = len(vectorizer.get_feature_names_out())

    # Load model
    nn_model = load_model(input_size)

    # Evaluate model with interpretability
    evaluate_model(nn_model, X_data, y_data)

if __name__ == "__main__":
    main()
