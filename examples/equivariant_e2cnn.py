import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from e2cnn import gspaces
from e2cnn import nn as enn  # Equivariant neural network module

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the G-CNN model
class GCNN(nn.Module):
    def __init__(self, n_classes=10):
        super(GCNN, self).__init__()
        
        # Define the rotation group: C4 (4 rotations: 0°, 90°, 180°, 270°)
        self.r2_act = gspaces.Rot2dOnR2(N=4)  # Cyclic group C4
        
        # Input field: scalar field (grayscale image, 1 channel)
        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # First layer: Equivariant convolution (8 feature maps)
        self.input_type = in_type
        out_type = enn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr])
        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=True)
        self.relu1 = enn.ReLU(out_type, inplace=True)
        self.pool1 = enn.PointwiseMaxPool(out_type, 2)  # 2x2 max pooling
        
        # Second layer: Another equivariant convolution (16 feature maps)
        in_type = out_type
        out_type = enn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        self.conv2 = enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=True)
        self.relu2 = enn.ReLU(out_type, inplace=True)
        self.pool2 = enn.PointwiseMaxPool(out_type, 2)  # 2x2 max pooling
        
        # Final layer: Group pooling + fully connected
        self.gpool = enn.GroupPooling(out_type)  # Pool over the group (rotation invariance)
        self.fc = nn.Linear(16 * 7 * 7, n_classes)  # After pooling: 16 channels, 7x7 spatial dim
        
    def forward(self, x):
        # Wrap input as a geometric tensor
        x = enn.GeometricTensor(x, self.input_type)
        
        # Forward pass through equivariant layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Group pooling to achieve invariance
        x = self.gpool(x)
        
        # Extract tensor and flatten for fully connected layer
        x = x.tensor
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x

# 2. Load and preprocess MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Initialize model, loss, and optimizer
model = GCNN(n_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 5. Test function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')

# 6. Run training and testing
for epoch in range(1, 6):  # 5 epochs for demo
    train(epoch)
    test()  