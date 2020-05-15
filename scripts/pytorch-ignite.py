"""
Based on
https://github.com/pytorch/ignite/blob/master/examples/contrib/mnist/mnist_with_tensorboard_logger.py
---

 MNIST example with training and validation monitoring using TensorboardX and Tensorboard.
 Requirements:
    Optionally TensorboardX (https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
    Tensorboard: `pip install tensorflow` (or just install tensorboard without the rest of tensorflow)
 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=/tmp/tensorboard_logs/
    ```
    Run the example:
    ```bash
    python mnist_with_tensorboard_logger.py --log_dir=/tmp/tensorboard_logs
    ```
"""
import sys
from argparse import ArgumentParser
import logging

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.engines.common import setup_common_training_handlers, setup_tb_logging, \
    add_early_stopping_by_val_score

LOG_INTERVAL = 10
HEAVY_LOG_INTERVAL = 100


# From https://tinyurl.com/pytorch-fast-mnist
class FastMNIST(MNIST):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_data_loaders(train_batch_size, val_batch_size, device):
    train_loader = DataLoader(
        FastMNIST(device, download=True, root="data", train=True), batch_size=train_batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        FastMNIST(device, download=False, root="data", train=False), batch_size=val_batch_size,
        shuffle=False
    )
    return train_loader, val_loader


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_dir):
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size, device)
    model = Net()

    model.to(device)  # Move model before creating optimizer
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        train_evaluator.run(train_loader)
        validation_evaluator.run(val_loader)

    setup_common_training_handlers(trainer, log_every_iters=LOG_INTERVAL)

    tb_logger = TensorboardLogger(log_dir=log_dir)

    def global_step_transform(_, __):
        return trainer.state.iteration

    # Compared to the default tb_logger behavior it is better to log everything using a single step counter
    # and log epoch numbers separately.
    tb_logger.attach(
        trainer,
        log_handler=lambda engine, logger, event_name: logger.writer.add_scalar("epoch", engine.state.epoch,
                                                                                engine.state.iteration),
        event_name=Events.ITERATION_COMPLETED(every=LOG_INTERVAL),
    )

    # Log trainer metrics
    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="training", output_transform=lambda loss: {"batchloss": loss}, metric_names="all",
            global_step_transform=global_step_transform
        ),
        event_name=Events.ITERATION_COMPLETED(every=LOG_INTERVAL),
    )

    # Log train evaluator metrics.
    tb_logger.attach(
        train_evaluator,
        log_handler=OutputHandler(tag="training", metric_names="all", global_step_transform=global_step_transform),
        event_name=Events.EPOCH_COMPLETED,
    )

    # Log validation evaluator metrics.
    tb_logger.attach(
        validation_evaluator,
        log_handler=OutputHandler(tag="validation", metric_names="all", global_step_transform=global_step_transform),
        event_name=Events.EPOCH_COMPLETED,
    )

    # Log optimizer metrics.
    tb_logger.attach(
        trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_COMPLETED(every=100)
    )

    # Log weights and gradients.
    tb_logger.attach(trainer, log_handler=WeightsScalarHandler(model),
                     event_name=Events.ITERATION_COMPLETED(every=HEAVY_LOG_INTERVAL))
    tb_logger.attach(trainer, log_handler=WeightsHistHandler(model),
                     event_name=Events.ITERATION_COMPLETED(every=HEAVY_LOG_INTERVAL))
    tb_logger.attach(trainer, log_handler=GradsScalarHandler(model),
                     event_name=Events.ITERATION_COMPLETED(every=HEAVY_LOG_INTERVAL))
    tb_logger.attach(trainer, log_handler=GradsHistHandler(model),
                     event_name=Events.ITERATION_COMPLETED(every=HEAVY_LOG_INTERVAL))

    # Add early stopping
    add_early_stopping_by_val_score(3, train_evaluator, trainer, "accuracy")

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)
    tb_logger.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument(
        "--val_batch_size", type=int, default=1000, help="input batch size for validation (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument(
        "--log_dir", type=str, default=None, help="log directory for Tensorboard log output"
    )

    args = parser.parse_args()

    # Setup engine logger
    logger = logging.getLogger("ignite.engine.engine.Engine")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_dir)
