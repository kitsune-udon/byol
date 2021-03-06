import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import STL10

from argparse_utils import from_argparse_args


def stl10_collate_fn(batch):
    images, labels = list(zip(*batch))
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


class STL10DataModule(pl.LightningDataModule):
    def __init__(self, dataset_root="./dataset",
                 train_batch_size=8, val_batch_size=8, num_workers=0):
        super().__init__()

        self.dataset_root = dataset_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

    def prepare_data(self, *args, **kwargs):
        STL10(self.dataset_root, download=True)

    def train_dataloader(self, *args, **kwargs):
        stl10_train = STL10(self.dataset_root, split='train+unlabeled',
                            download=False, transform=self.transform)

        return DataLoader(stl10_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          batch_size=self.train_batch_size,
                          collate_fn=stl10_collate_fn)

    def val_dataloader(self, *args, **kwargs):
        stl10_val = STL10(self.dataset_root, split='test',
                          download=False, transform=self.transform)

        return DataLoader(stl10_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          batch_size=self.val_batch_size,
                          collate_fn=stl10_collate_fn)

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
