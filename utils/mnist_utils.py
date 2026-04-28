import pickle
import numpy as np


def load_mnist_from_pkl(filepath="mnist.pkl", normalize=True, standardize=False):

    with open(filepath, "rb") as f:
        data = pickle.load(f, encoding='latin1')

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    if x_train.dtype != np.float32:
        x_train = x_train.astype(np.float32)
        x_val = x_val.astype(np.float32)
        x_test = x_test.astype(np.float32)

    if standardize:
        train_mean = np.mean(x_train)
        train_std = np.std(x_train)
        epsilon = 1e-8

        x_train = (x_train - train_mean) / (train_std + epsilon)
        x_val = (x_val - train_mean) / (train_std + epsilon)
        x_test = (x_test - train_mean) / (train_std + epsilon)

    elif normalize:
        tr_max = np.max(x_train)
        x_train = x_train /tr_max
        x_val = x_val /tr_max
        x_test = x_test /tr_max

    if len(x_train.shape) == 2 and x_train.shape[1] == 784:
        x_train = x_train.reshape(-1, 28, 28)
        x_val = x_val.reshape(-1, 28, 28)
        x_test = x_test.reshape(-1, 28, 28)

    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)

    assert x_train.shape[1:] == (28, 28), f"Expected x_train to be 28x28, got {x_train.shape[1:]}"
    assert x_val.shape[1:] == (28, 28), f"Expected x_val to be 28x28, got {x_val.shape[1:]}"
    assert x_test.shape[1:] == (28, 28), f"Expected x_test to be 28x28, got {x_test.shape[1:]}"

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
