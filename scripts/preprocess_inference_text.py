import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

# Paths
INFERENCE_DATA_PATH = "data/inference_text/synthetic_McDonald_s_Reviews.csv"
VECTORIZER_PATH = "models/text_vectorizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
PREPROCESSED_X_PATH = "data/inference_text/X_inference.npy"
PREPROCESSED_Y_PATH = "data/inference_text/y_inference.npy"

def preprocess_text_inference(data_path, vectorizer_path, label_encoder_path):
    """
    Preprocess text data for inference.

    Args:
        data_path (str): Path to the inference text dataset (CSV).
        vectorizer_path (str): Path to the trained vectorizer.
        label_encoder_path (str): Path to the trained label encoder.

    Returns:
        np.ndarray: Preprocessed features.
        np.ndarray: Labels (if available, else None).
    """
    print("Loading inference data...")
    df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Check for text and target columns
    if "review" not in df.columns:
        raise ValueError("Column 'review' not found in the dataset!")
    reviews = df["review"]

    y = None
    if "rating" in df.columns:
        print("Encoding labels...")
        label_encoder = joblib.load(label_encoder_path)
        df["sentiment"] = df["rating"].apply(lambda x: "positive" if "4" in x or "5" in x else "negative")
        y = label_encoder.transform(df["sentiment"])

    print("Loading vectorizer...")
    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform(reviews)

    return X, y

def main():
    # Preprocess inference data
    print("Preprocessing inference text data...")
    X_inference, y_inference = preprocess_text_inference(
        INFERENCE_DATA_PATH,
        VECTORIZER_PATH,
        LABEL_ENCODER_PATH
    )

    # Save preprocessed data
    print(f"Saving preprocessed data to {PREPROCESSED_X_PATH} and {PREPROCESSED_Y_PATH}...")
    np.save(PREPROCESSED_X_PATH, X_inference.toarray())
    if y_inference is not None:
        np.save(PREPROCESSED_Y_PATH, y_inference)

    print("Inference text data preprocessing complete.")

if __name__ == "__main__":
    main()
