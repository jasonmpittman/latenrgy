import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_DIR = "data/inference_image"
LABELS = ["hot_dog", "not_hot_dog"]  # Update with actual class names
IMAGE_SIZE = (64, 64)  # Resize all images to this size (consistent with training)

def load_and_preprocess_images(data_dir, labels, image_size):
    """
    Load and preprocess images for inference.

    Args:
        data_dir (str): Path to the directory containing labeled image subdirectories.
        labels (list): List of label subdirectory names.
        image_size (tuple): Target image size (width, height) for resizing.

    Returns:
        tuple: (images, encoded_labels)
            - images: NumPy array of preprocessed image data.
            - encoded_labels: NumPy array of encoded labels (if labels are available).
    """
    images = []
    image_labels = []

    # Process each label directory
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)

            try:
                # Load and preprocess the image
                img = Image.open(file_path).convert("RGB")
                img = img.resize(image_size)
                img_array = np.asarray(img) / 255.0  # Normalize to [0, 1]
                images.append(img_array)
                image_labels.append(label)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(image_labels)

    return np.array(images), np.array(encoded_labels)

def main():
    # Preprocess the inference images
    print("Loading and preprocessing inference images...")
    images, labels = load_and_preprocess_images(DATA_DIR, LABELS, IMAGE_SIZE)

    # Save the preprocessed data for inference scripts
    np.save(os.path.join(DATA_DIR, "images.npy"), images)
    np.save(os.path.join(DATA_DIR, "labels.npy"), labels)
    print(f"Preprocessed images saved to {DATA_DIR}/images.npy")
    print(f"Labels saved to {DATA_DIR}/labels.npy")

if __name__ == "__main__":
    main()
