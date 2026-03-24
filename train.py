import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import zipfile
import os

import mlflow
import mlflow.keras

# =========================
# MLflow setup
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Assignment5_Classifier")

# =========================
# Unzip dataset
# =========================
zip_path = "mnist_csv.zip"

if not os.path.exists("mnist_data"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("mnist_data")

# =========================
# Load data
# =========================
data = pd.read_csv("mnist_data/mnist_train.csv")

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Normalize
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

# Split
X_train, X_test = X[:50000], X[50000:]
y_train, y_test = y[:50000], y[50000:]

# =========================
# Model
# =========================
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Training
# =========================
epochs = 3  # enough to get >0.85 accuracy

with mlflow.start_run() as run:

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Get real accuracy
    final_accuracy = float(history.history["val_accuracy"][-1])

    print(f"Final Accuracy: {final_accuracy}")

    # Log metric
    mlflow.log_metric("final_accuracy", final_accuracy)

    # Save model
    mlflow.keras.log_model(model, "model")

    # Save run_id (CRITICAL)
    run_id = run.info.run_id

    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print("Saved model_info.txt")

print("Training finished.")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
#
# import torchvision
# from torchvision import transforms
#
# import mlflow
# import mlflow.pytorch
#
#
#
# transform_pipeline = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# train_data = torchvision.datasets.MNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=transform_pipeline
# )
#
# test_data = torchvision.datasets.MNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=transform_pipeline
# )
#
#
#
# class DigitClassifier(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#         self.layer1 = nn.Linear(784, 128)
#         self.activation = nn.ReLU()
#         self.layer2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#
#         x = x.view(x.size(0), -1)
#
#         x = self.layer1(x)
#         x = self.activation(x)
#         x = self.layer2(x)
#
#         return x
#
#
#
# def run_training(lr, batch, num_epochs):
#
#     train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=batch)
#
#     net = DigitClassifier()
#
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr=lr)
#
#     mlflow.set_experiment("Assignment3_AhmedMohammed")
#
#     with mlflow.start_run():
#
#         mlflow.log_param("learning_rate", lr)
#         mlflow.log_param("batch_size", batch)
#         mlflow.log_param("epochs", num_epochs)
#
#         mlflow.set_tag("student_name", "Ahmed Mohammed")
#         mlflow.set_tag("student_id", "202202168")
#
#         for epoch in range(num_epochs):
#
#             net.train()
#
#             total_loss = 0
#
#             for images, labels in train_loader:
#
#                 optimizer.zero_grad()
#
#                 predictions = net(images)
#
#                 loss = loss_function(predictions, labels)
#
#                 loss.backward()
#
#                 optimizer.step()
#
#                 total_loss += loss.item()
#
#             avg_loss = total_loss / len(train_loader)
#
#             net.eval()
#
#             correct = 0
#             samples = 0
#
#             with torch.no_grad():
#
#                 for images, labels in test_loader:
#
#                     outputs = net(images)
#
#                     _, predicted = torch.max(outputs, 1)
#
#                     samples += labels.size(0)
#
#                     correct += (predicted == labels).sum().item()
#
#             accuracy = correct / samples
#
#             print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
#
#             mlflow.log_metric("loss", avg_loss, step=epoch)
#             mlflow.log_metric("accuracy", accuracy, step=epoch)
#
#         mlflow.pytorch.log_model(net, "mnist_model")
#         # Save run_id
#         run_id = run.info.run_id
#
#         with open("model_info.txt", "w") as f:
#             f.write(run_id)
#
#         print("Saved model_info.txt")
#
#
# run_training(0.1, 64, 5)
# run_training(0.01, 64, 5)
# run_training(0.001, 64, 5)
# run_training(0.001, 32, 5)
# run_training(0.0005, 128, 5)