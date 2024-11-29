import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocess_images import X_train_torch, y_train_torch, X_test_torch, y_test_torch
from utils import save_results

# Neural Network Model
class HotDogNet(nn.Module):
    def __init__(self):
        super(HotDogNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 31 * 31, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        print(f"Shape before flattening: {x.shape}")  # Debugging line
        x = x.contiguous().view(x.size(0), -1)  # Ensure the tensor is contiguous before flattening
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure the models directory exists
os.makedirs("../../models", exist_ok=True)

# Start energy tracker
tracker = EmissionsTracker()
tracker.start()

# Model, Loss, Optimizer
model = HotDogNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Neural Network
start_time = time.time()
epochs = 10
batch_size = 32

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i in range(0, len(X_train_torch), batch_size):
        X_batch = X_train_torch[i:i + batch_size].to(device)
        y_batch = y_train_torch[i:i + batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

training_time = time.time() - start_time

# Stop energy tracker
emissions = tracker.stop()

# Save Model
model_path = "models/nn_image_model.pth"
print(f"Saving model to: {os.path.abspath(model_path)}")  # Debugging line
torch.save(model.state_dict(), model_path)

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X_test_torch.to(device)).cpu().numpy().flatten()
    y_pred_class = (y_pred > 0.5).astype(int)

metrics = {
    "Accuracy": accuracy_score(y_test_torch.numpy(), y_pred_class),
    "Precision": precision_score(y_test_torch.numpy(), y_pred_class),
    "Recall": recall_score(y_test_torch.numpy(), y_pred_class),
    "F1 Score": f1_score(y_test_torch.numpy(), y_pred_class),
    "ROC-AUC": roc_auc_score(y_test_torch.numpy(), y_pred),
}

# Save results
save_results("nn_image", metrics, training_time, emissions)
