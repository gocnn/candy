import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define modern LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 32x32x1 -> 32x32x6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)             # 32x32x6 -> 28x28x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                   # 2x2 max pooling
        self.fc1 = nn.Linear(16 * 6 * 6, 120)                              # 6x6x16 = 576 -> 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))      # 32x32x6
        x = self.pool(x)                  # 16x16x6
        x = self.relu(self.conv2(x))      # 12x12x16
        x = self.pool(x)                  # 6x6x16
        x = x.view(-1, 16 * 6 * 6)       # Flatten to 576
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)                   # Output logits
        return x

# Data preparation
transform = transforms.Compose([
    transforms.Pad(2),  # Pad to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model, loss function, and optimizer
print("cuda is available:", torch.cuda.is_available())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()  # Use nn.MSELoss() with one-hot labels if needed
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # loss = criterion(outputs, labels_one_hot)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if (batch_idx + 1) % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100 * correct / total
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
            
    return running_loss / len(train_loader)

# Testing function
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_accuracy = test(model, test_loader, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Save model
torch.save(model.state_dict(), 'lenet.pth')