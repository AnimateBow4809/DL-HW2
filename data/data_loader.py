import numpy as np
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = self.x.reshape(-1, 784)

    def __len__(self) -> int:
        return len(self.x)

