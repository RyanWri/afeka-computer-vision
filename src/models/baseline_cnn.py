import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            32 * 48 * 48, 1
        )  # Input is flattened size, output is a single neuron

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Conv + ReLU
        x = self.pool(x)  # MaxPooling
        x = self.flatten(x)  # Flatten
        x = self.fc(x)
        return x
