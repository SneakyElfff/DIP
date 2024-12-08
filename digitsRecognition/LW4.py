import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from matplotlib import pyplot as plt
from torch import optim

# Constants
BATCH_SIZE = 64
NUM_EPOCHS = 3

# Device configuration
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Data transformation
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

# Datasets and Dataloaders
train_validation_data = MNIST('.', train=True, transform=transform, download=True)
test_data = MNIST('.', train=False, transform=transform, download=True)

train_data, validation_data = random_split(train_validation_data, [50000, 10000])

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Sample visualization
X, y = next(iter(train_dataloader))

plt.figure(figsize=(12, 5))
for i in range(15):
    x, y = next(iter(train_dataloader))
    plt.subplot(3, 5, i + 1)
    plt.title(f'{y[0]}')
    plt.imshow(x[0].squeeze(), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

# Model definition
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.model(x)

# Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
model = MyModel().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

# Training function
def train_model(model, train_loader, val_loader, loss_function, optimizer, device, epochs):
    train_loss_hist, val_loss_hist, val_acc_hist = [], [], []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_function(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss_hist.append(total_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, correct_predictions, total_samples = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_function(pred, y).item()
                correct_predictions += (pred.argmax(1) == y).sum().item()
                total_samples += y.size(0)
        val_loss_hist.append(val_loss / len(val_loader))
        val_acc_hist.append(100 * correct_predictions / total_samples)

        # Output progress
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss_hist[-1]:.4f}")
        print(f"Validation Loss: {val_loss_hist[-1]:.4f}, Validation Accuracy: {val_acc_hist[-1]:.2f}%")

    return train_loss_hist, val_loss_hist, val_acc_hist

# Save model function
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load model function
def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")

# Training or Loading
model_path = "mnist_model.pth"
try:
    load_model(model, model_path)
except FileNotFoundError:
    print("Model not found. Starting training...")
    train_loss_hist, val_loss_hist, val_acc_hist = train_model(
        model, train_dataloader, validation_dataloader, loss_function, optimizer, device, NUM_EPOCHS
    )
    save_model(model, model_path)

# Testing
test_loss, test_accuracy = 0, 0
model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_function(pred, y).item()
        test_accuracy += (pred.argmax(1) == y).sum().item()

test_loss /= len(test_dataloader)
test_accuracy = 100 * test_accuracy / len(test_dataloader.dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")