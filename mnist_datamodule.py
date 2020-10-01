import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from argparse_utils import from_argparse_args


def mnist_collate_fn(batch):
    images, labels = list(zip(*batch))
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, dataset_root="./dataset", train_batch_size=8, val_batch_size=8, num_workers=0):
        super().__init__()
        self.dataset_root = dataset_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.137,), (0.3081,))
        ])
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def prepare_data(self, *args, **kwargs):
        MNIST(self.dataset_root, download=True)

    def train_dataloader(self, *args, **kwargs):
        mnist_train = MNIST(self.dataset_root, train=True,
                            download=False, transform=self.transform)
        return DataLoader(mnist_train, shuffle=True, num_workers=self.num_workers,
                          batch_size=self.train_batch_size, collate_fn=mnist_collate_fn)

    def val_dataloader(self, *args, **kwargs):
        mnist_val = MNIST(self.dataset_root, train=False,
                          download=False, transform=self.transform)
        return DataLoader(mnist_val, shuffle=False, num_workers=self.num_workers,
                          batch_size=self.val_batch_size, collate_fn=mnist_collate_fn)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--train_batch_size", type=int,
                            default=64, help="batch size for training")
        parser.add_argument("--val_batch_size", type=int,
                            default=64, help="batch size for validation")
        parser.add_argument("--num_workers", type=int, default=4,
                            help="number of processes for dataloader")
        parser.add_argument("--dataset_root", type=str,
                            default="./dataset", help="root path of dataset")
        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)
