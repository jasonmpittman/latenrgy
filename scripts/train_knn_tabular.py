import os
import joblib
import time
from codecarbon import EmissionsTracker
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess_tabular import X_train, X_test, y_train, y_test
from utils import save_results

# Ensure the models directory exists
os.makedirs("../../models", exist_ok=True)

# Start energy tracker
tracker = EmissionsTracker(allow_multiple_runs=True, output_file=f"emissions_knn_tabular.csv")
tracker.start()

# Train k-NN
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5)  # You can tune 'n_neighbors' based on experimentation
knn.fit(X_train, y_train)
training_time = time.time() - start_time

# Stop energy tracker
emissions = tracker.stop()

# Save model
model_path = "models/knn_tabular_model.pkl"
print(f"Saving model to: {os.path.abspath(model_path)}")
joblib.dump(knn, model_path)

# Evaluate
predictions = knn.predict(X_test)
metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, average="weighted"),
    "Recall": recall_score(y_test, predictions, average="weighted"),
    "F1 Score": f1_score(y_test, predictions, average="weighted"),
}

# Save results
save_results("knn_tabular", metrics, training_time, emissions)

# Print results for debugging
print(f"Training Time: {training_time:.2f} seconds")
print(f"Energy Consumption: {emissions:.2f} kWh")
print("Metrics:", metrics)
