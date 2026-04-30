import torch
import yaml

from data.data_loader import MnistDataset
from utils.mnist_utils import *


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_datasets(config):
    if config['data']['dataset'] == "mnist":
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_from_pkl(filepath=config['data']['dataset_path'])
    elif config['data']['dataset'] == "fashion":
        (x_train_full, y_train_full), (x_test, y_test) = load_fashion_mnist_raw(data_dir=config['data']['dataset_path'])
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
    train_dataset = MnistDataset(x_train, y_train)
    val_dataset = MnistDataset(x_val, y_val)
    test_dataset = MnistDataset(x_test, y_test)
    return train_dataset, val_dataset, test_dataset
