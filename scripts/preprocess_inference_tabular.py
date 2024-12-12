import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Paths
INFERENCE_DATA_PATH = "data/inference_tabular/synthetic_fitness_class.csv"
PREPROCESSED_X_PATH = "data/inference_tabular/X_inference.npy"
PREPROCESSED_Y_PATH = "data/inference_tabular/y_inference.npy"
SELECTED_FEATURES = [0, 1, 2, 3, 4]  # Replace with actual indices from training

# Load dataset
def load_tabular_data(file_path):
    """
    Load the tabular dataset from a CSV file.

    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Dataset as a DataFrame.
    """
    df = pd.read_csv(file_path)

    # Clean and convert 'days_before' to numeric
    if "days_before" in df.columns:
        if df["days_before"].dtype == "object":
            df["days_before"] = df["days_before"].str.replace(" days", "").astype(float)
        else:
            df["days_before"] = df["days_before"].astype(float)

    return df


# Preprocess inference data
def preprocess_inference_data(df):
    """
    Preprocess the inference dataset.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        np.ndarray: Preprocessed data ready for inference.
    """
    # Drop non-predictive column
    if "booking_id" in df.columns:
        df = df.drop(columns=["booking_id"])

    # Target variable
    y = df["attended"] if "attended" in df.columns else None
    X = df.drop(columns=["attended"]) if "attended" in df.columns else df

    # Define preprocessing steps
    numeric_features = ["months_as_member", "weight", "days_before"]
    categorical_features = ["day_of_week", "time", "category"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Apply preprocessing
    X_preprocessed = preprocessor.fit_transform(X)

    # Select features based on training
    X_selected = X_preprocessed[:, SELECTED_FEATURES]

    return X_selected, y

def main():
    # Load synthetic inference data
    print("Loading inference data...")
    df = load_tabular_data(INFERENCE_DATA_PATH)

    # Preprocess data
    print("Preprocessing inference data...")
    X_inference, y_inference = preprocess_inference_data(df)

    # Save preprocessed data
    print(f"Saving preprocessed data to {PREPROCESSED_X_PATH} and {PREPROCESSED_Y_PATH}...")
    np.save(PREPROCESSED_X_PATH, X_inference)
    if y_inference is not None:
        np.save(PREPROCESSED_Y_PATH, y_inference)

    print("Inference data preprocessing complete.")

if __name__ == "__main__":
    main()
