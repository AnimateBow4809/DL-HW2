from collections import OrderedDict

import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    implementation of residual block using original resnet paper
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main_path = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ("bn1", nn.BatchNorm2d(out_channels)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv2", nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ("bn2", nn.BatchNorm2d(out_channels))
        ]))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        out += identity
        out = self.final_relu(out)
        return out


class Bottleneck(nn.Module):
    """
    implementation of bottleneck block using resnetB from bags of tricks paper
    """
    def __init__(self, in_channels, intermediate_channels, out_channels, stride=1):
        super().__init__()
        self.main_path = nn.Sequential(OrderedDict([
            ("conv1",
             nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            ("bn1", nn.BatchNorm2d(intermediate_channels)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv2",
             nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ("bn2", nn.BatchNorm2d(intermediate_channels)),
            ("relu2", nn.ReLU(inplace=True)),
            ("conv3",
             nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            ("bn3", nn.BatchNorm2d(out_channels)),
        ]))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        out += identity
        out = self.final_relu(out)
        return out
