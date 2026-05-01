import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from data.data_loader import MnistDataset
from models.inception_model import InceptionModel
from models.residual_model import ResidualModel
from models.resnext_model import ResNeXtModel
from utils.trainer import Trainer

from utils.config_utils import load_config, get_datasets
from utils.mnist_utils import load_mnist_from_pkl, load_fashion_mnist_raw

if __name__ == '__main__':
    config = load_config('../config/config.yaml')
    device = config['model']['device']
    train_dataset, val_dataset, test_dataset = get_datasets(config)
    batch_size = config['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = config['model']
    arch = config['model']['arch']
    if arch == 'a':
        model = ResidualModel()
    elif arch == 'b':
        model = InceptionModel()
    elif arch == 'c':
        model = ResNeXtModel()
    else:
        raise ValueError("Unknown arch {}".format(arch))
    model = model.to(device)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=config['optimizer']['lr'],
                                momentum=config['optimizer']['momentum'])
    trainer = Trainer(
        device=device,
        model=model,
        optimizer=optimizer,
        loss_fn=CrossEntropyLoss(),
    )

    trainer.train(train_loader, val_loader, use_early_stopping=config['model']['early_stopping'],
                  epochs=config['model']['epochs'])
    trainer.plot_learning_curves()
    trainer.plot_confusion_matrix(val_loader)
    trainer.save_model(f"../models/saved/mnist_model_{arch}.pth")
