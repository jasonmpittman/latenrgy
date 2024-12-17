import os
import shap
import numpy as np
import joblib
import torch
import torch.nn as nn
import time
import argparse
from multiprocessing import Pool
from joblib import Parallel, delayed

# Paths
DATA_DIR = "data/inference_text/"
MODELS_DIR = "models/"
SHAP_OUTPUT_DIR = "data/inference_text/shap_values/"

# Ensure SHAP output directory exists
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

class TextNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 2)  # Matches the trained model
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model(model_type):
    """
    Load the trained model based on model type.
    """
    if model_type == "SVM":
        return joblib.load(os.path.join(MODELS_DIR, "svm_text_model.pkl"))
    elif model_type == "k-NN":
        return joblib.load(os.path.join(MODELS_DIR, "knn_text_model.pkl"))
    elif model_type == "Random Forest":
        return joblib.load(os.path.join(MODELS_DIR, "rf_text_model.pkl"))
    elif model_type == "Neural Network":
        model = TextNN(input_size=5000, num_classes=2)  # Update input size to match training
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "nn_text_model.pth")))
        model.eval()  # Set the model to evaluation mode
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def dynamic_sampling(X_data, nsamples_factor=10):
    """
    Dynamically calculate the number of samples based on the number of features.
    """
    num_features = X_data.shape[1]
    return min(1000, num_features * nsamples_factor)  # Limit to 1000 samples

def compute_chunk_shap_values(chunk, model):
    """
    Compute SHAP values for a specific data chunk using TreeExplainer.
    Additivity check is disabled at shap_values call level.
    """
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="interventional"
    )
    return explainer.shap_values(chunk, check_additivity=False)

def compute_shap_values_with_multiprocessing_tree(model, X_data, subset_size=50, num_chunks=4):
    """
    Compute SHAP values for Random Forest using multiprocessing.
    """
    X_subset = X_data[:subset_size]

    # Verify data shape
    if hasattr(model, "n_features_in_"):
        print(f"Model trained on {model.n_features_in_} features, input data has {X_subset.shape[1]} features.")
        if X_subset.shape[1] != model.n_features_in_:
            raise ValueError("Mismatch between training data shape and input data shape for SHAP computation.")

    # Divide data into chunks
    chunks = np.array_split(X_subset, num_chunks)

    # Use multiprocessing to compute SHAP values for each chunk
    with Pool(processes=num_chunks) as pool:
        shap_values = pool.starmap(compute_chunk_shap_values, [(chunk, model) for chunk in chunks])

    # Flatten the list of shap_values
    shap_values = np.concatenate(shap_values, axis=0)
    return shap_values

def compute_shap_values(model, X_data, model_type, subset_size=50, nsamples=100, n_jobs=-1):
    """
    Compute SHAP values for the given model and data using batching and multiprocessing where applicable.
    """
    X_subset = X_data[:subset_size]

    if model_type in ["k-NN"]:
        print(f"Using KernelExplainer for {model_type}...")
        explainer = shap.KernelExplainer(model.predict, X_subset)
        shap_values = explainer.shap_values(X_data, nsamples=nsamples)

    elif model_type == "Random Forest":
        print(f"Using TreeExplainer for {model_type} with multiprocessing...")
        shap_values = compute_shap_values_with_multiprocessing_tree(
            model, X_data, subset_size=subset_size, num_chunks=os.cpu_count()
        )

    elif model_type == "Neural Network":
        print(f"Using DeepExplainer with GPU batching for {model_type}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        X_subset_tensor = torch.tensor(X_subset, dtype=torch.float32).to(device)
        explainer = shap.DeepExplainer(model, X_subset_tensor)

        # Batch processing for SHAP values
        batch_size = 10
        shap_values = []
        for i in range(0, len(X_data), batch_size):
            batch = X_data[i:i + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            shap_values_batch = explainer.shap_values(batch_tensor)
            shap_values.append(shap_values_batch)

        shap_values = np.concatenate(shap_values, axis=0)

    elif model_type == "SVM":
        print(f"Using KernelExplainer for {model_type} with batching and multiprocessing...")

        # Use a subset of data as the background dataset for KernelExplainer
        X_subset = X_data[:subset_size]

        # Divide the dataset into batches
        batch_size = 50  # Adjust based on available memory and runtime
        batches = [X_data[i:i + batch_size] for i in range(0, len(X_data), batch_size)]
        num_batches = len(batches)
        print(f"Divided data into {num_batches} batches of size {batch_size}.")

        # Use multiprocessing to compute SHAP values for each batch
        with Pool(processes=4) as pool:  # Adjust 'processes' to match available CPU cores
            shap_values_batches = pool.starmap(compute_shap_batch, [(batch, model, X_subset, nsamples) for batch in batches])

        # Combine the SHAP values from all batches
        shap_values = np.concatenate(shap_values_batches, axis=0)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return shap_values

# Define the batch processing function
def compute_shap_batch(batch, model, X_subset, nsamples):
    """
    Compute SHAP values for a single batch using KernelExplainer.
    """
    explainer = shap.KernelExplainer(model.predict, X_subset)
    return explainer.shap_values(batch, nsamples=nsamples)


def precompute_shap(model_type, dataset="text", subset_size=50, nsamples=100):
    """
    Precompute SHAP values for a specific model type and dataset.
    """
    # Load the preprocessed inference data
    print(f"Loading preprocessed inference data for {dataset} dataset...")
    X_data = np.load(os.path.join(DATA_DIR, "X_inference.npy"))

    # Load the model
    print(f"Loading {model_type} model...")
    model = load_model(model_type)

    # Compute SHAP values
    print(f"Computing SHAP values for {model_type} on {dataset} dataset...")
    start_time = time.time()
    shap_values = compute_shap_values(model, X_data, model_type, subset_size=subset_size, nsamples=nsamples)
    elapsed_time = time.time() - start_time

    # Save SHAP values
    shap_output_path = os.path.join(SHAP_OUTPUT_DIR, f"shap_values_{model_type.lower()}_{dataset}.npy")
    np.save(shap_output_path, shap_values)
    print(f"SHAP values saved to {shap_output_path}")
    print(f"SHAP computation completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Precompute SHAP values for a specific model.")
    parser.add_argument("--model", type=str, required=True, help="Model type (e.g., SVM, k-NN, Random Forest, Neural Network)")
    args = parser.parse_args()

    # Precompute SHAP for the specified model
    precompute_shap(model_type=args.model)
