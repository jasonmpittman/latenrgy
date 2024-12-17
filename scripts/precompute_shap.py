import os
import argparse
import numpy as np
import shap
import torch
import torch.nn as nn
from multiprocessing import Pool

# Constants for paths
DATA_DIRS = {
    "text": "data/inference_text/",
    "tabular": "data/inference_tabular/",
    "image": "data/inference_image/"
}
MODELS_DIR = "models/"

SHAP_OUTPUT_DIRS = {
    "text": "data/inference_text/shap_values/",
    "tabular": "data/inference_tabular/shap_values/",
    "image": "data/inference_image/shap_values/"
}

# Ensure SHAP output directory exists
def ensure_shap_output_dir(dataset):
    shap_dir = SHAP_OUTPUT_DIRS[dataset]
    os.makedirs(shap_dir, exist_ok=True)
    return shap_dir

# Model Architecture for Neural Networks
class TextNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TabularNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TabularNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Dropout for training

    def forward(self, x, shap_mode=False):
        x = self.fc1(x)
        x = self.relu(x)
        if not shap_mode:  # Disable dropout during SHAP
            x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class ImageNN(nn.Module):
    def __init__(self, num_classes):
        super(ImageNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 32 * 32, 128)  # Adjust input size after conv layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Conv + ReLU + Pool
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Load Model Function
def load_model(model_type, dataset, X_data=None):
    if model_type in ["SVM", "k-NN", "RandomForest"]:
        import joblib
        model_file = os.path.join(MODELS_DIR, f"{model_type.lower()}_{dataset}_model.pkl")
        return joblib.load(model_file)
    elif model_type == "NeuralNetwork":
        if dataset == "tabular":
            input_size = X_data.shape[1]  # Dynamically determine input size
            num_classes = 2  # Binary classification
            model = TabularNN(input_size, num_classes)
            model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"nn_tabular_model.pth")))
            model.eval()
            return model
        elif dataset == "image":
            num_classes = 2  # Binary classification
            model = ImageNN(num_classes)
            model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"nn_image_model.pth")))
            model.eval()
            return model
        else:
            raise ValueError(f"Unsupported dataset for NeuralNetwork: {dataset}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Batch SHAP computation for SVM and k-NN
def compute_shap_batch(batch, model, X_subset, nsamples):
    explainer = shap.KernelExplainer(model.predict, X_subset)
    return explainer.shap_values(batch, nsamples=nsamples)

# Main SHAP Computation Function
def compute_shap_values(model, X_data, model_type, dataset, subset_size=50, nsamples=100):
    X_subset = X_data[:subset_size]

    if model_type in ["SVM", "k-NN"]:
        print(f"Using KernelExplainer for {model_type} on {dataset} data...")
        batches = [X_data[i:i + 50] for i in range(0, len(X_data), 50)]
        with Pool(processes=4) as pool:
            shap_values_batches = pool.starmap(compute_shap_batch, [(batch, model, X_subset, nsamples) for batch in batches])
        shap_values = np.concatenate(shap_values_batches, axis=0)

    elif model_type == "RandomForest":
        print(f"Using TreeExplainer for {model_type} on {dataset} data...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)

    elif model_type == "NeuralNetwork":
        print(f"Using GradientExplainer for {model_type} on {dataset} data...")
        
        # Ensure device is set
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # Set model to evaluation mode

        # Convert X_data to PyTorch tensors
        X_data_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
        X_subset_tensor = torch.tensor(X_data[:subset_size], dtype=torch.float32).to(device)

        # Define a custom forward function for SHAP
        def forward_function(x):
            with torch.no_grad():  # Disable gradients
                return model(x)

        # Use GradientExplainer instead of DeepExplainer
        explainer = shap.GradientExplainer(forward_function, X_subset_tensor)

        # Compute SHAP values
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(X_data_tensor)

        print("SHAP computation completed.")


    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return shap_values

# Main Precompute Function
def precompute_shap(model_type, dataset, subset_size=50, nsamples=100):
    print(f"Loading preprocessed inference data for {dataset} dataset...")
    X_data_path = os.path.join(DATA_DIRS[dataset], "X_inference.npy" if dataset != "image" else "images.npy")
    X_data = np.load(X_data_path)

    print(f"Loading {model_type} model for {dataset} dataset...")
    model = load_model(model_type, dataset, X_data=X_data)

    print(f"Precomputing SHAP values for {model_type} on {dataset} dataset...")
    shap_values = compute_shap_values(model, X_data, model_type, dataset, subset_size, nsamples)

    shap_output_dir = ensure_shap_output_dir(dataset)
    shap_output_path = os.path.join(shap_output_dir, f"shap_values_{model_type.lower()}_{dataset}.npy")
    np.save(shap_output_path, shap_values)
    print(f"SHAP values for {model_type} saved to {shap_output_path}")


# Argument Parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute SHAP values for different models and datasets.")
    parser.add_argument("--model", type=str, required=True, help="Model type: SVM, k-NN, RandomForest, NeuralNetwork")
    parser.add_argument("--dataset", type=str, required=True, choices=["text", "tabular", "image"],
                        help="Dataset type: text, tabular, image")
    parser.add_argument("--subset_size", type=int, default=50, help="Subset size for SHAP KernelExplainer")
    parser.add_argument("--nsamples", type=int, default=100, help="Number of samples for SHAP KernelExplainer")
    args = parser.parse_args()

    precompute_shap(args.model, args.dataset, subset_size=args.subset_size, nsamples=args.nsamples)
