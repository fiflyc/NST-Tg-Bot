import torch
import torch.nn as nn


class InverseNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        x0 = self.upsample0(self.conv0(x))
        x1 = self.upsample1(self.conv1(x0))
        x2 = self.conv2(x1)

        return x2