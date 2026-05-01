import torch
import torchvision.transforms as transforms
import yaml

from data.data_loader import MnistDataset
from utils.mnist_utils import *


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_datasets(config):
    if config['data']['dataset'] == "mnist":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            # transforms.Normalize((0.1307,), (0.3081,)) # Already normalized
        ])
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_from_pkl(
            filepath=config['data']['mnist_path'])

    elif config['data']['dataset'] == "fashion":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        (x_train_full, y_train_full), (x_test, y_test) = load_fashion_mnist_raw(data_dir=config['data']['fashion_path'])
        total_size = len(x_train_full)
        val_size = 10000
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_size, generator=generator).numpy()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        x_train, y_train = x_train_full[train_indices], y_train_full[train_indices]
        x_val, y_val = x_train_full[val_indices], y_train_full[val_indices]

    else:
        raise ValueError("Unknown dataset")

    train_dataset = MnistDataset(x_train, y_train, transform=train_transform)
    val_dataset = MnistDataset(x_val, y_val, transform=val_test_transform)
    test_dataset = MnistDataset(x_test, y_test, transform=val_test_transform)
    return train_dataset, val_dataset, test_dataset
