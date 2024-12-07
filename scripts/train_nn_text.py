import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess_text import X_train_vec, y_train, X_test_vec, y_test
from utils import save_results

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train_vec, dtype=torch.float32).to(device)
X_test_torch = torch.tensor(X_test_vec, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_torch = torch.tensor(y_test, dtype=torch.long).to(device)

# Neural Network model
class TextClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model configuration
input_size = X_train_torch.shape[1]
num_classes = len(set(y_train))
model = TextClassifier(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Ensure the models directory exists
os.makedirs("../../models", exist_ok=True)

# Start energy tracker
tracker = EmissionsTracker(allow_multiple_runs=True, output_file=f"emissions_nn_text.csv")
tracker.start()

# Train the model
start_time = time.time()
epochs = 10
batch_size = 32

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i in range(0, X_train_torch.size(0), batch_size):
        X_batch = X_train_torch[i:i + batch_size]
        y_batch = y_train_torch[i:i + batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

training_time = time.time() - start_time

# Stop energy tracker
emissions = tracker.stop()

# Save the model
model_path = "models/nn_text_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {os.path.abspath(model_path)}")

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_torch)
    _, predictions = torch.max(outputs, 1)
    predictions = predictions.cpu().numpy()

metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, average="weighted"),
    "Recall": recall_score(y_test, predictions, average="weighted"),
    "F1 Score": f1_score(y_test, predictions, average="weighted"),
}

# Save results
save_results("nn_text", metrics, training_time, emissions)

# Print results for debugging
print(f"Training Time: {training_time:.2f} seconds")
print(f"Energy Consumption: {emissions:.2f} kWh")
print("Metrics:", metrics)
