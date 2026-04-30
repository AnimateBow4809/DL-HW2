import torch
from torch.utils.data import Dataset
import numpy as np


class MnistDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)
            image = image.unsqueeze(0)
        label = torch.tensor(label, dtype=torch.int64)
        return image, label
