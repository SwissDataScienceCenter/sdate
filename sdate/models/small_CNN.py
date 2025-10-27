import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN_small(nn.Module):
    def __init__(self, output_dim):
        super(CNN_small, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Output: 16x16 (if input is 32x32)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # Output: 8x8
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: 4x4
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # Output: 2x2
        self.bn4 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(16384, output_dim)  # Adjusted to match output size after pooling

        self.weight_init(self.fc)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    def forward(self, image):
        x = F.relu(self.bn1(self.conv1(image)))  # 16x16
        x = F.relu(self.bn2(self.conv2(x)))  # 8x8
        x = F.relu(self.bn3(self.conv3(x)))  # 4x4
        x = F.relu(self.bn4(self.conv4(x)))  # 2x2

        x = F.max_pool2d(x, 2)  # Output: 1x1 (assuming 2x2 input)
        x = torch.flatten(x, 1)  # Flatten to 64 features
        x = self.fc(x)  # Output layer

        return x  # Output (x, y) coordinates