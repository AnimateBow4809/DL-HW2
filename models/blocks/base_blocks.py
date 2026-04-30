from torch import nn


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, use_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.use_relu:
            x = self.relu(x)
        return x
