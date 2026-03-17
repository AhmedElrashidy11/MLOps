import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import mlflow
import mlflow.pytorch



transform_pipeline = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform_pipeline
)

test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform_pipeline
)



class DigitClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 128)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)

        return x



def run_training(lr, batch, num_epochs):

    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch)

    net = DigitClassifier()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    mlflow.set_experiment("Assignment3_AhmedMohammed")

    with mlflow.start_run():

        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch)
        mlflow.log_param("epochs", num_epochs)

        mlflow.set_tag("student_name", "Ahmed Mohammed")
        mlflow.set_tag("student_id", "202202168")

        for epoch in range(num_epochs):

            net.train()

            total_loss = 0

            for images, labels in train_loader:

                optimizer.zero_grad()

                predictions = net(images)

                loss = loss_function(predictions, labels)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            net.eval()

            correct = 0
            samples = 0

            with torch.no_grad():

                for images, labels in test_loader:

                    outputs = net(images)

                    _, predicted = torch.max(outputs, 1)

                    samples += labels.size(0)

                    correct += (predicted == labels).sum().item()

            accuracy = correct / samples

            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

        mlflow.pytorch.log_model(net, "mnist_model")



run_training(0.1, 64, 5)
run_training(0.01, 64, 5)
run_training(0.001, 64, 5)
run_training(0.001, 32, 5)
run_training(0.0005, 128, 5)