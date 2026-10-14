"""The example's model: a small CNN that labels 32×32 grey scans as healthy or disease.

In your own project this is your model code; the connector imports it from the project folder.
"""
import torch
import torch.nn as nn


class ScanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 8 * 8, 32)
        self.relu3 = nn.ReLU()
        self.head = nn.Linear(32, 2)  # logits for (healthy, disease)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.head(self.relu3(self.fc1(torch.flatten(x, 1))))
