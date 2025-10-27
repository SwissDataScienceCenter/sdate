import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

class ResnetGrayscale(nn.Module):
    def __init__(self, pretrained, out_features=1, only_fc=False):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_features)
        if only_fc:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.resnet(x.repeat(1, 3, 1, 1))