import torch
import torch.nn as nn

from models.blocks.base_blocks import BasicConv2d
from models.blocks.resnext_blocks import ResNeXtBottleneck


class ResNeXtModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNeXtModel, self).__init__()
        self.block1 = BasicConv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.block2 = BasicConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, pooling="max")
        self.block3 = ResNeXtBottleneck(in_channels=64, out_channels=64, cardinality=8, group_width=4)
        self.block4 = ResNeXtBottleneck(in_channels=64, out_channels=128, cardinality=8, group_width=8, pooling="max")
        self.block5 = BasicConv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, pooling="max")
        self.block6 = ResNeXtBottleneck(in_channels=256, out_channels=256, cardinality=8, group_width=16, pooling="avg")
        self.fc = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
