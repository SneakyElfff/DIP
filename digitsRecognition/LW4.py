import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch import optim
from tabulate import tabulate

BATCH_SIZE = 64
NUM_EPOCHS = 3

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def prepare_dataloaders(batch_size):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])

    # загрузить MNIST-датасеты
    mnist_train_validation = MNIST('.', train=True, transform=transform, download=True)
    mnist_test = MNIST('.', train=False, transform=transform, download=True)

    train_size, validation_size = 50000, 10000
    train_dataset, validation_dataset = random_split(mnist_train_validation, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, validation_loader, test_loader


# архитектура сверточной нейросети
class ModelArchitecture(nn.Module):
    def __init__(self):
        super(ModelArchitecture, self).__init__()
        self.layers = nn.Sequential(
            # первый сверточный блок
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # второй сверточный блок
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Flatten for fully connected layers
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, train_loader, validation_loader, loss_function, optimizer, device, num_epochs):
    train_losses = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs), 'started...')
        # фаза обучения
        model.train()
        total_train_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_data)
            loss = loss_function(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # фаза валидации
        model.eval()
        total_validation_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for valid_data, valid_labels in validation_loader:
                valid_data, valid_labels = valid_data.to(device), valid_labels.to(device)

                predictions = model(valid_data)
                total_validation_loss += loss_function(predictions, valid_labels).item()
                correct_predictions += (predictions.argmax(dim=1) == valid_labels).sum().item()
                total_samples += valid_labels.size(0)

        avg_validation_loss = total_validation_loss / len(validation_loader)
        validation_accuracy = 100 * correct_predictions / total_samples

        validation_losses.append(avg_validation_loss)
        validation_accuracies.append(validation_accuracy)

    table_data = [
        [epoch + 1, train_losses[epoch], validation_losses[epoch], validation_accuracies[epoch]]
        for epoch in range(num_epochs)
    ]

    headers = ["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    return train_losses, validation_losses, validation_accuracies


def test_model(model, test_loader, loss_function, device):
    model.eval()
    total_test_loss = 0
    total_correct_predictions = 0
    with torch.no_grad():
        for test_data, test_labels in test_loader:
            test_data, test_labels = test_data.to(device), test_labels.to(device)
            predictions = model(test_data)

            total_test_loss += loss_function(predictions, test_labels).item()
            total_correct_predictions += (predictions.argmax(dim=1) == test_labels).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy_percentage = 100 * total_correct_predictions / len(test_loader.dataset)

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy_percentage:.2f}%")


def main():
    train_loader, validation_loader, test_loader = prepare_dataloaders(BATCH_SIZE)

    model = ModelArchitecture().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    model_save_path = "mnist_cnn_model.pth"
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()
        print("Model's been successfully loaded.")
    except FileNotFoundError:
        print("Training a new model...")
        train_model(
            model, train_loader, validation_loader, loss_function, optimizer, device, NUM_EPOCHS
        )
        torch.save(model.state_dict(), model_save_path)
        print("Model's been trained and saved.")

    test_model(model, test_loader, loss_function, device)


if __name__ == "__main__":
    main()