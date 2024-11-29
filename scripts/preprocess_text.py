import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Path to the data
CSV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../latenrgy/data/text/McDonald_s_Reviews.csv"))

# Load the CSV file with ISO-8859-1 encoding
def load_text_data(csv_file):
    """
    Load and preprocess text data from the given CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing preprocessed data.
    """
    try:
        # Read the file with the correct encoding
        df = pd.read_csv(csv_file, encoding="ISO-8859-1")
    except Exception as e:
        raise ValueError(f"Error loading the file: {e}")

    # Ensure required fields exist
    if "review" not in df or "rating" not in df:
        raise ValueError("CSV file must contain 'review' and 'rating' columns.")

    # Map ratings to sentiment (positive or negative)
    df["sentiment"] = df["rating"].apply(lambda x: "positive" if "4" in x or "5" in x else "negative")

    return df

# Load data
df = load_text_data(CSV_FILE)

# Split into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Encode labels as integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Save the label mapping for reference
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Debug shapes
print(f"X_train shape: {X_train_vec.shape}, y_train shape: {len(y_train)}")
print(f"X_test shape: {X_test_vec.shape}, y_test shape: {len(y_test)}")