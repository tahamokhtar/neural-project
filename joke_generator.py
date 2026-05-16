# =========================================
# Fashion-MNIST using MLP - PyTorch
# =========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================================
# 1) DATA
# =========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# =========================================
# 2) MODEL
# =========================================

class MLP(nn.Module):

    def __init__(self, h1=128, h2=64):
        super().__init__()

        self.fc1 = nn.Linear(784, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.view(-1, 784)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# =========================================
# 3) TRAIN FUNCTION
# =========================================

def train(model, lr, epochs):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    accuracies = []

    for epoch in range(epochs):

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        losses.append(avg_loss)
        accuracies.append(acc)

        print(f"Epoch {epoch+1}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {acc:.2f}%")
        print("---------------------")

    return losses, accuracies

# =========================================
# 4) TEST FUNCTION
# =========================================

def test(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"\nTest Accuracy = {accuracy:.2f}%")

    return accuracy

# =========================================
# 5) EXPERIMENT 1
# =========================================

print("\n===== Experiment 1 =====")

model1 = MLP(h1=128, h2=64)

loss1, acc1 = train(
    model1,
    lr=0.001,
    epochs=10
)

test1 = test(model1)

# =========================================
# 6) EXPERIMENT 2
# =========================================

print("\n===== Experiment 2 =====")

model2 = MLP(h1=256, h2=128)

loss2, acc2 = train(
    model2,
    lr=0.0005,
    epochs=10
)

test2 = test(model2)

# =========================================
# 7) RESULTS
# =========================================

print("\n===== FINAL RESULTS =====")

print(f"Experiment 1 Accuracy: {test1:.2f}%")
print(f"Experiment 2 Accuracy: {test2:.2f}%")

# =========================================
# 8) VISUALIZATION
# =========================================

# Loss Curve
plt.plot(loss1, label="Exp 1")
plt.plot(loss2, label="Exp 2")

plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Accuracy Curve
plt.plot(acc1, label="Exp 1")
plt.plot(acc2, label="Exp 2")

plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
