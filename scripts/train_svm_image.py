import os
import joblib
import time
from codecarbon import EmissionsTracker
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess_images import X_train, y_train, X_test, y_test
from utils import save_results

# Flatten data for SVM
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Ensure the models directory exists
os.makedirs("../../models", exist_ok=True)

# Start energy tracker
tracker = EmissionsTracker()
tracker.start()

# Train SVM
start_time = time.time()
svm = SVC(probability=True, kernel="linear", random_state=42)
svm.fit(X_train_flat, y_train)
training_time = time.time() - start_time

# Stop energy tracker
emissions = tracker.stop()

# Save model
model_path = "models/svm_image_model.pkl"
print(f"Saving model to: {os.path.abspath(model_path)}")  # Debugging line
joblib.dump(svm, model_path)

# Evaluate
predictions = svm.predict(X_test_flat)
metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, average="weighted"),
    "Recall": recall_score(y_test, predictions, average="weighted"),
    "F1 Score": f1_score(y_test, predictions, average="weighted"),
}

# Save results
save_results("svm_image", metrics, training_time, emissions)
