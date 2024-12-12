import os
import joblib
import time
from codecarbon import EmissionsTracker
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from utils import save_results

# Paths
X_TRAIN_PATH = "data/text/X_train.npy"
Y_TRAIN_PATH = "data/text/y_train.npy"
MODEL_PATH = "models/knn_text_model.pkl"

# Ensure the models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    X = np.load(X_TRAIN_PATH)
    y = np.load(Y_TRAIN_PATH)

    # Split into training and validation sets
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start energy tracker
    tracker = EmissionsTracker(allow_multiple_runs=True, output_file="emissions/emissions_knn_text.csv")
    tracker.start()

    # Train k-NN
    print("Training k-NN model...")
    start_time = time.time()  # Start timing
    knn = KNeighborsClassifier(n_neighbors=5)  # You can tune 'n_neighbors' as needed
    knn.fit(X_train, y_train)
    training_time = time.time() - start_time  # Calculate training time

    # Stop energy tracker
    emissions = tracker.stop()

    # Save the trained model
    print(f"Saving model to: {os.path.abspath(MODEL_PATH)}")
    joblib.dump(knn, MODEL_PATH)

    # Evaluate the model
    print("Evaluating model...")
    predictions = knn.predict(X_val)
    metrics = {
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(y_val, predictions, average="weighted"),
        "Recall": recall_score(y_val, predictions, average="weighted"),
        "F1 Score": f1_score(y_val, predictions, average="weighted"),
    }

    # Save results
    save_results("knn_text", metrics, training_time, emissions)

    # Print results for debugging
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Energy Consumption: {emissions:.2f} kWh")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
