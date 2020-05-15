import os
from typing import Union, List, Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl


class MNISTModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super().__init__()

        # get hyperparams, etc...
        self.hparams = hparams

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # called with self(x)
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

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_pred = y_hat.argmax(dim=1, keepdim=False)
        correct = (y_pred == y).sum().type(torch.float)
        total = y.shape[0]
        acc = correct/total

        tensorboard_logs = dict(train_loss=loss, train_acc=acc)
        return dict(loss=loss, acc=acc, log=tensorboard_logs)

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)

        y_pred = y_hat.argmax(dim=1, keepdim=False)
        correct = (y_pred == y).sum().type(torch.float)
        total = y.shape[0]

        return dict(val_loss=val_loss, val_acc=correct/total)

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = dict(val_loss=avg_loss, val_acc=avg_acc)
        return dict(val_loss=avg_loss, log=tensorboard_logs)

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)

        y_pred = y_hat.argmax(dim=1, keepdim=False)
        correct = (y_pred == y).sum().type(torch.float)
        total = y.shape[0]

        return dict(test_loss=F.cross_entropy(y_hat, y), test_acc=correct/total)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        test_tensorboard_logs = dict(test_loss=avg_loss, test_acc=avg_acc)
        self.logger.log_hyperparams(None, test_tensorboard_logs)
        return dict(test_loss=avg_loss, log=test_tensorboard_logs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        self.mnist_train = MNIST("data", train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST("data", train=False, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        loader = DataLoader(self.mnist_train, batch_size=32, num_workers=0)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.mnist_test, batch_size=32, num_workers=0)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.mnist_test, batch_size=32, num_workers=0)
        return loader


if __name__ == '__main__':
    mnist_model = MNISTModel(dict(batch_size=32))

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(gpus=1, fast_dev_run=True, early_stop_callback=True)
    trainer.fit(mnist_model)
    trainer.test(mnist_model)