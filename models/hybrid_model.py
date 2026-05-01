import torch
import torch.nn as nn
from models.blocks.base_blocks import BasicConv2d
from models.blocks.inception_blocks import InceptionBlock
from models.blocks.residual_blocks import ResidualBlock
from models.blocks.resnext_blocks import ResNeXtBottleneck


class HybridModel(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridModel, self).__init__()

        self.block1 = BasicConv2d(1, 32, kernel_size=3, padding=1)
        self.block2 = BasicConv2d(32, 64, kernel_size=3, padding=1, pooling="max")
        self.block3 = ResidualBlock(64, 64)
        self.block4 = ResidualBlock(64, 128,pooling="max")
        self.block5 = BasicConv2d(128, 256, kernel_size=3, padding=1, pooling="max")
        self.block6 = InceptionBlock(in_channels=256, branch1_out=32, branch2_out=64, branch3_out=96, branch4_out=64,
                                     pooling="avg")
        self.fc = nn.Linear(256, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.softmax(x) handled by the loss
        return x
