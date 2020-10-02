from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from byol import TargetNetworkUpdator


def main(dm_cls, model_cls, model_args, logger_name):
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = dm_cls.add_argparse_args(parser)
    parser = model_cls.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    dm = dm_cls.from_argparse_args(args)

    byol_args = model_cls.extract_kwargs_from_argparse_args(
        args, **model_args)
    model = model_cls(**byol_args)

    logger = TensorBoardLogger('tb_logs', name=logger_name)
    logger.log_hyperparams(args)

    checkpoint = ModelCheckpoint(
        monitor='val_acc', filepath=None, save_top_k=1)

    trainer = pl.Trainer.from_argparse_args(args,
                                            deterministic=True,
                                            callbacks=[TargetNetworkUpdator()],
                                            checkpoint_callback=checkpoint,
                                            logger=logger
                                            )

    trainer.fit(model, dm)
