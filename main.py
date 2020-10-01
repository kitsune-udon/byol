from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from byol import BYOL, TargetNetworkUpdator, default_augmentation
from mnist_datamodule import MNISTDataModule


def get_base_model():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(9216, 128)
    )

def get_byol_params():
    return {
        "base_model": get_base_model(),
        "augmentation": default_augmentation(28, is_color_image=False),
        "projector_isize": 128,
        "projector_hsize": 256,
        "projector_osize": 128,
        "predictor_hsize": 256
        }

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MNISTDataModule.add_argparse_args(parser)
    parser = BYOL.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)
    
    seed_everything(args.seed)
    dm = MNISTDataModule.from_argparse_args(args)
    byol_params = BYOL.extract_kwargs_from_argparse_args(args, **get_byol_params())
    model = BYOL(**byol_params)
    logger = TensorBoardLogger('tb_logs', name='byol_mnist')
    logger.log_hyperparams(args)
    checkpoint = ModelCheckpoint(monitor='val_acc', filepath=None, save_top_k=1)
    trainer = pl.Trainer.from_argparse_args(args,
                                            deterministic=True,
                                            callbacks=[TargetNetworkUpdator()],
                                            checkpoint_callback=checkpoint,
                                            logger=logger
                                            )
    trainer.fit(model, dm)

if __name__ == '__main__':
    main()
