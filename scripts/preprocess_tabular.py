import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# File path
CSV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../latenrgy/data/tabular/fitness_class_2212.csv"))

# Load dataset
def load_tabular_data(file_path):
    """
    Load the fitness class dataset from a CSV file.

    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Dataset as a DataFrame.
    """
    df = pd.read_csv(file_path)

    # Clean and convert 'days_before' to integer
    if "days_before" in df.columns:
        df["days_before"] = df["days_before"].str.replace(" days", "").astype(float)
    
    return df

# Preprocess data
def preprocess_data_with_selection(df):
    """
    Preprocess and perform feature selection on the dataset.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        tuple: Preprocessed training and testing data (X_train, X_test, y_train, y_test).
    """
    # Drop non-predictive column
    df = df.drop(columns=["booking_id"])

    # Target variable
    y = df["attended"]
    X = df.drop(columns=["attended"])

    # Define preprocessing steps
    numeric_features = ["months_as_member", "weight", "days_before"]
    categorical_features = ["day_of_week", "time", "category"]

    # Pipelines for preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # Handle NaNs in numerical data
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle NaNs in categorical data
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply preprocessing
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Feature selection using Sequential Feature Selector
    sfs = SFS(RandomForestClassifier(n_estimators=100, random_state=42),
              k_features=5,  # Select top 5 features
              forward=True,
              floating=False,
              scoring='accuracy',
              cv=5)

    # Fit SFS on training data
    sfs.fit(X_train_preprocessed, y_train)

    # Get selected feature indices
    selected_features_indices = list(sfs.k_feature_idx_)
    print("Selected feature indices:", selected_features_indices)

    # Transform datasets to include only selected features
    X_train_selected = X_train_preprocessed[:, selected_features_indices].toarray()
    X_test_selected = X_test_preprocessed[:, selected_features_indices].toarray()

    return X_train_selected, X_test_selected, y_train, y_test


    return X_train_selected, X_test_selected, y_train, y_test

# Load and preprocess data
df = load_tabular_data(CSV_FILE)
X_train, X_test, y_train, y_test = preprocess_data_with_selection(df)

# Debug: Output shapes
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")