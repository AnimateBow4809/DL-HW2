import torch


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
    for name, module in model.named_modules():
        if not name.startswith('fc') and isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
