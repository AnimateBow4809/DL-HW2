import numpy as np

from utils.mnist_utils import load_mnist_from_pkl


class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MnistDataset(Dataset):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = self.x.reshape(-1, 784)

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[start_idx: start_idx + self.batch_size]
            batch_x = self.dataset.x[batch_indices]
            batch_y = self.dataset.y[batch_indices]
            yield batch_x, batch_y

