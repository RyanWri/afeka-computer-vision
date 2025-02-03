import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Input is flattened size, output is a single neuron
        self.fc = nn.Linear(32, 1)

    def forward(self, x, return_features: bool):
        x = F.relu(self.conv1(x))  # Conv + ReLU
        x = self.pool(x)  # MaxPooling
        x = self.global_avg_pool(x)  # Reduce feature maps to (batch_size, 32, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to shape (batch_size, 32)
        # return_features = True -> feature embedding
        # return_features = False -> classification output
        return x if return_features else self.fc(x)
