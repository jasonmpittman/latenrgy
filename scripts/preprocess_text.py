import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

# Paths
TRAINING_DATA_PATH = "data/text/McDonald_s_Reviews.csv"
PREPROCESSED_X_PATH = "data/text/X_train.npy"
PREPROCESSED_Y_PATH = "data/text/y_train.npy"
VECTORIZER_PATH = "models/text_vectorizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

def preprocess_text(data_path, vectorizer_path, label_encoder_path):
    """
    Preprocess text data for training.

    Args:
        data_path (str): Path to the training text dataset (CSV).
        vectorizer_path (str): Path to save the trained vectorizer.
        label_encoder_path (str): Path to save the trained label encoder.

    Returns:
        np.ndarray: Preprocessed features.
        np.ndarray: Encoded labels.
    """
    print("Loading training data...")
    df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Check for required columns
    if "review" not in df.columns or "rating" not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'rating' columns!")

    # Convert ratings into sentiments
    print("Converting ratings into sentiments...")
    df["sentiment"] = df["rating"].apply(lambda x: "positive" if "4" in x or "5" in x else "negative")
    reviews = df["review"]
    sentiments = df["sentiment"]

    # Create and fit TfidfVectorizer
    print("Fitting TfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(reviews)

    # Create and fit LabelEncoder
    print("Fitting LabelEncoder...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(sentiments)

    # Save the vectorizer and label encoder
    print("Saving vectorizer and label encoder...")
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, label_encoder_path)

    return X, y

def main():
    # Preprocess training data
    print("Preprocessing training text data...")
    X_train, y_train = preprocess_text(
        TRAINING_DATA_PATH,
        VECTORIZER_PATH,
        LABEL_ENCODER_PATH
    )

    # Save preprocessed data
    print(f"Saving preprocessed data to {PREPROCESSED_X_PATH} and {PREPROCESSED_Y_PATH}...")
    np.save(PREPROCESSED_X_PATH, X_train.toarray())
    np.save(PREPROCESSED_Y_PATH, y_train)

    print("Training text data preprocessing complete.")

if __name__ == "__main__":
    main()
