import os
import joblib
import time
from codecarbon import EmissionsTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess_text import X_train_vec, y_train, X_test_vec, y_test
from utils import save_results

# Ensure the models directory exists
os.makedirs("../../models", exist_ok=True)

# Start energy tracker
tracker = EmissionsTracker()
tracker.start()

# Train Random Forest
start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust 'n_estimators' as needed
rf.fit(X_train_vec, y_train)
training_time = time.time() - start_time

# Stop energy tracker
emissions = tracker.stop()

# Save model
model_path = "models/rf_text_model.pkl"
print(f"Saving model to: {os.path.abspath(model_path)}")
joblib.dump(rf, model_path)

# Evaluate
predictions = rf.predict(X_test_vec)
metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, average="weighted"),
    "Recall": recall_score(y_test, predictions, average="weighted"),
    "F1 Score": f1_score(y_test, predictions, average="weighted"),
}

# Save results
save_results("rf_text", metrics, training_time, emissions)

# Print results for debugging
print(f"Training Time: {training_time:.2f} seconds")
print(f"Energy Consumption: {emissions:.2f} kWh")
print("Metrics:", metrics)
