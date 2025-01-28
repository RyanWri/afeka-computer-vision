import torch.nn as nn
import torch.nn.functional as F


class SingleConvFeatureExtractor(nn.Module):
    def __init__(
        self, input_channels=3, output_channels=32, kernel_size=3, stride=1, padding=1
    ):
        super(SingleConvFeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Optional: reduces feature map size

    def forward(self, x):
        x = F.relu(self.conv(x))  # Apply convolution + ReLU
        x = self.pool(x)  # Downsample (optional)
        return x
