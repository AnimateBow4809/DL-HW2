import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from data.data_loader import MnistDataset
from models.inception_model import InceptionModel
from models.residual_model import ResidualModel
from utils.Trainer import Trainer
from utils.mnist_utils import load_mnist_from_pkl


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    config = load_config('../config/config.yaml')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_from_pkl(filepath=config['data']['filepath'])

    train_dataset = MnistDataset(x_train, y_train)
    val_dataset = MnistDataset(x_val, y_val)
    test_dataset = MnistDataset(x_test, y_test)

    batch_size = config['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = config['model']
    if config['model']['arch'] == 'a':
        model = ResidualModel()
    elif config['model']['arch'] == 'a':
        model = InceptionModel()
    else:
        model = InceptionModel()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['optimizer']['lr'],
                                momentum=config['optimizer']['momentum'])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=CrossEntropyLoss(),
    )

    trainer.train(train_loader, val_loader, use_early_stopping=config['model']['early_stopping'],
                  epochs=config['model']['epochs'])
    trainer.plot_learning_curves()
    trainer.plot_confusion_matrix(val_loader)
    trainer.save_model("../models/saved/mnist_model.pkl")
