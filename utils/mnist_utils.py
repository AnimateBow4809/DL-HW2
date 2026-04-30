import pickle
import os
import gzip
import numpy as np


def load_mnist_from_pkl(filepath="mnist.pkl"):
    with open(filepath, "rb") as f:
        data = pickle.load(f, encoding='latin1')

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    x_train = x_train.reshape(-1, 28, 28)
    x_val = x_val.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_fashion_mnist_raw(data_dir="data/dataset/fashion"):
    train_img_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_lbl_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_img_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_lbl_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

    def read_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 28, 28).copy()

    def read_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data.copy()

    x_train_full = read_images(train_img_path)
    y_train_full = read_labels(train_lbl_path)

    x_test = read_images(test_img_path)
    y_test = read_labels(test_lbl_path)

    return (x_train_full, y_train_full), (x_test, y_test)