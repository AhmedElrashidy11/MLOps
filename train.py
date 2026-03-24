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
# # --------------------
# # Data
# # --------------------
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# train_data = torchvision.datasets.MNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=transform
# )
#
# test_data = torchvision.datasets.MNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=transform
# )
#
# # --------------------
# # Model
# # --------------------
# class DigitClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# # --------------------
# # Training Function
# # --------------------
# def run_training(lr=0.001, batch_size=64, epochs=3):
#
#     mlflow.set_tracking_uri("file:./mlruns")
#     mlflow.set_experiment("Assignment5")
#
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=batch_size)
#
#     model = DigitClassifier()
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     with mlflow.start_run() as run:
#
#         mlflow.log_param("lr", lr)
#         mlflow.log_param("batch_size", batch_size)
#         mlflow.log_param("epochs", epochs)
#
#         final_accuracy = 0.0
#
#         for epoch in range(epochs):
#
#             # TRAIN
#             model.train()
#             total_loss = 0
#
#             for x, y in train_loader:
#                 optimizer.zero_grad()
#                 out = model(x)
#                 loss = loss_fn(out, y)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#
#             avg_loss = total_loss / len(train_loader)
#
#             # EVAL
#             model.eval()
#             correct = 0
#             total = 0
#
#             with torch.no_grad():
#                 for x, y in test_loader:
#                     out = model(x)
#                     _, pred = torch.max(out, 1)
#                     total += y.size(0)
#                     correct += (pred == y).sum().item()
#
#             accuracy = correct / total
#             final_accuracy = accuracy
#
#             print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
#
#             mlflow.log_metric("loss", avg_loss, step=epoch)
#             mlflow.log_metric("accuracy", accuracy, step=epoch)
#
#         # FINAL METRIC (IMPORTANT FOR DEPLOY JOB)
#         mlflow.log_metric("final_accuracy", final_accuracy)
#
#         # Save model
#         mlflow.pytorch.log_model(model, "model")
#
#         # Save run id for CI
#         run_id = run.info.run_id
#         with open("model_info.txt", "w") as f:
#             f.write(run_id)
#
#         print("Saved Run ID:", run_id)
#
#
# if __name__ == "__main__":
#     run_training(lr=0.001)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import mlflow
import mlflow.pytorch

# -------------------
# Data
# -------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# -------------------
# Model
# -------------------
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------
# Training
# -------------------
def run_training(lr=0.001, batch_size=64, epochs=3):

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Assignment5")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = DigitClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run() as run:

        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        final_accuracy = 0.0

        for epoch in range(epochs):

            # TRAIN
            model.train()
            total_loss = 0

            for x, y in train_loader:
                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # EVAL
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for x, y in test_loader:
                    out = model(x)
                    _, pred = torch.max(out, 1)
                    total += y.size(0)
                    correct += (pred == y).sum().item()

            accuracy = correct / total
            final_accuracy = accuracy

            print(f"Epoch {epoch+1} | Loss={avg_loss:.4f} | Acc={accuracy:.4f}")

            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

        # FINAL METRIC (IMPORTANT)
        mlflow.log_metric("final_accuracy", final_accuracy)

        # Save model
        mlflow.pytorch.log_model(model, "model")

        # Save run id
        run_id = run.info.run_id
        with open("model_info.txt", "w") as f:
            f.write(run_id)

        print("Run ID saved:", run_id)


if __name__ == "__main__":
    run_training(lr=0.001)
