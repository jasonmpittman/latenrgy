import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import torch

# Use absolute path for data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../latenrgy/data/image"))
IMG_SIZE = (64, 64)

def load_images_from_directory(directory):
    """
    Loads images from a directory where each subdirectory represents a class.
    
    Args:
        directory (str): Path to the parent directory containing subdirectories for each class.
    
    Returns:
        np.array: Array of images.
        np.array: Array of corresponding labels.
    """
    images, labels = [], []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue  # Skip non-directory files
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            if os.path.isfile(file_path):  # Ensure it's a file
                img = cv2.imread(file_path)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)  # Resize image to uniform dimensions
                    images.append(img)
                    labels.append(label)  # Use subdirectory name as label
    return np.array(images), np.array(labels)

# Load train and test datasets
train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

X_train, y_train = load_images_from_directory(train_dir)
X_test, y_test = load_images_from_directory(test_dir)

# Normalize image data to range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Encode labels as integers (e.g., hot_dog -> 0, not_hot_dog -> 1)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Save the label mapping for reference
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Convert datasets to PyTorch tensors for the neural network
X_train_torch = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
X_test_torch = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Make labels 2D
y_test_torch = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Make labels 2D

# Debug: Print dataset shapes
print(f"X_train shape: {X_train_torch.shape}, y_train shape: {y_train_torch.shape}")
print(f"X_test shape: {X_test_torch.shape}, y_test shape: {y_test_torch.shape}")
