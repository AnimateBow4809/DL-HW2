import yaml
from data.data_loader import MnistDataset, DataLoader
from utils.Trainer import Trainer
from utils.loss import CrossEntropyLoss
from utils.mnist_utils import load_mnist_from_pkl
from utils.optimizaer import *


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    config = load_config('../config/config.yaml')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_from_pkl(
        filepath=config['data']['filepath'],
        normalize=config['data']['normalize'],
        standardize=config['data']['standardize']
    )

    train_dataset = MnistDataset(x_train, y_train)
    val_dataset = MnistDataset(x_val, y_val)
    test_dataset = MnistDataset(x_test, y_test)

    batch_size = config['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = SGDMomentum if config['optimizer']['momentum'] else SGD
    trainer = Trainer(
        network_dims=config['model']['network_dims'],
        optimizer_instance=optimizer(learning_rate=config['optimizer']['learning_rate']
                                     ,l2_penalty=config['optimizer']['l2_penalty']),
        loss_function_class=CrossEntropyLoss,
        epochs=config['model']['epochs'],
        dense_layer_kwargs=config['model']['dense_layer_kwargs'],
    )

    trainer.train(train_loader, val_loader,use_early_stopping=config['model']['early_stopping'])
    trainer.plot_learning_curves()
    trainer.plot_confusion_matrix(val_loader)
    trainer.save_model("../models/saved/mnist_model.pkl")
