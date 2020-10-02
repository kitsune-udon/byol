import torch.nn as nn

from byol import BYOL, default_augmentation
from mnist_datamodule import MNISTDataModule
from train_utils import main


class BYOLMNIST(BYOL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = default_augmentation(28, is_color_image=False)
        self.online_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9216, 128)
        )
        self.copy()


byol_args = {
    "projector_isize": 128,
    "projector_hsize": 256,
    "projector_osize": 128,
    "predictor_hsize": 256
}


if __name__ == '__main__':
    main(MNISTDataModule, BYOLMNIST, byol_args, "byol_mnist")
