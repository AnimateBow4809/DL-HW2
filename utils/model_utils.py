import torch

from models.blocks.base_blocks import BasicConv2d


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
    for name, module in model.named_modules():
        if not name.startswith('fc') and isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def get_first_block_activation(model,input):
    block = model.block1
    if not isinstance(block, BasicConv2d):
        raise TypeError("First block is not a BasicConv2d")
    x = block.conv(input)
    y = block.bn(x)
    if block.use_relu:
        z = block.relu(y)
    else:
        z = y
    return x,y,z
