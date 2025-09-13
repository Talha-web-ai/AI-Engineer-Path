import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------
# Define CNN
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------
# Training function
# -----------------------
def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1, 2):  # just 1 epoch for demo
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")
        print(f"Epoch {epoch} Average loss: {total_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "models/cnn_model.pth")
    print("✅ Model saved to models/cnn_model.pth")

if __name__ == "__main__":
    train_model()
