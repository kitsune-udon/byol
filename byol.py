import copy
import math
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import (ColorJitter, RandomGrayscale,
                                 RandomHorizontalFlip, RandomResizedCrop,
                                 RandomSolarize)
from kornia.filters import GaussianBlur2d
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from torch.optim import AdamW

from argparse_utils import extract_kwargs_from_argparse_args


class EarlyStoppingWithSkip(EarlyStopping):
    def __init__(self, *args, **kwargs):
        self.n_epochs = kwargs.pop('n_epochs')
        super().__init__(*args, **kwargs)

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.n_epochs:
            pass
        else:
            self._run_early_stopping_check(trainer, pl_module)

class RandomApply(nn.Module):
    def __init__(self, proc, p):
        super().__init__()
        self.proc = proc 
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.proc(x)

def default_augmentation(image_size, is_color_image=True):
    def get_kernel_size(image_size):
        kernel_size = image_size // 10
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        kernel_size = (kernel_size, kernel_size)
        return kernel_size

    def color_model(gaussian_p, solarize_p):
        return nn.Sequential(
                RandomResizedCrop((image_size, image_size), interpolation="BICUBIC"),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
                RandomApply(GaussianBlur2d(get_kernel_size(image_size), (0.1, 2.0)), p=gaussian_p),
                RandomSolarize(0, 0, p=solarize_p)
        )
    
    def monochrome_model(gaussian_p, solarize_p):
        return nn.Sequential(
                RandomResizedCrop((image_size, image_size), interpolation="BICUBIC"),
                RandomHorizontalFlip(),
                RandomApply(GaussianBlur2d(get_kernel_size(image_size), (0.1, 2.0)), p=gaussian_p),
                RandomSolarize(0, 0, p=solarize_p)
        )

    if is_color_image:
        return [color_model(*ps) for ps in [(1., 0.), (0.1, 0.2)]]
    else:
        return [monochrome_model(*ps) for ps in [(1., 0.), (0.1, 0.2)]]

class TargetNetworkUpdator(pl.Callback):
    def __init__(self, tau_base=0.996):
        super().__init__()
        self.tau_base = tau_base
    
    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        reference_model, target_model = pl_module.online_encoder, pl_module.target_encoder
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        progress = pl_module.global_step / max_steps
        tau = 1 - (1 - self.tau_base) * math.cos(math.pi * progress) * 0.5
        for ref_params, target_params in zip(reference_model.parameters(), target_model.parameters()):
            target_params.data = ref_params.data * (1 - tau) + target_params.data * tau

class MLP(nn.Module):
    def __init__(self, isize, hsize, osize):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(isize, hsize),
            nn.BatchNorm1d(hsize),
            nn.ReLU(),
            nn.Linear(hsize, osize)
        )

    def forward(self, x):
        return self.net(x)

class BYOL(pl.LightningModule):
    def __init__(self,
                 base_model=None,
                 augmentation=None,
                 projector_isize=None,
                 projector_hsize=None,
                 projector_osize=None,
                 predictor_hsize=None,
                 learning_rate=1e-3,
                 weight_decay=1.5e-6,
                 warmup_epochs=10,
                 max_epochs=1000
                 ):
        super().__init__()
        self.save_hyperparameters()
        assert(base_model is not None)
        assert(isinstance(base_model, nn.Module))
        assert(augmentation is not None)

        self.augment = augmentation

        projector = MLP(projector_isize, projector_hsize, projector_osize)
        self.predictor = MLP(projector_osize, predictor_hsize, projector_osize)
        self.online_encoder = nn.Sequential(base_model, projector)
        self.target_encoder = copy.deepcopy(self.online_encoder)
    
    def forward(self, x):
        return self.online_encoder[0](x)
    
    def calc_loss(self, x):
        def proc(x, y):
            out_online = self.predictor(self.online_encoder(x))
            with torch.no_grad():
                out_target = self.target_encoder(y)
            return -2 * F.cosine_similarity(out_online, out_target.detach(), dim=-1).mean()

        if isinstance(self.augment, nn.Module):
            imgs = [self.augment(x), self.augment(x)]
        elif isinstance(self.augment, (list, tuple)):
            imgs = [self.augment[0](x), self.augment[1](x)]
        else:
            raise TypeError("augmentation must be torch Module or its list")

        return proc(imgs[0], imgs[1]) + proc(imgs[1], imgs[0])

    def training_step(self, batch, batch_idx):
        x, _ = batch
        return self.calc_loss(x)

    def validation_step(self, batch, batch_idx):
        x, label = batch
        with torch.no_grad():
            y = self.online_encoder[0](x)
        return {'representation': y, 'label': label}

    def validation_epoch_end(self, outputs):
        ys = torch.cat([o['representation'] for o in outputs]).cpu().numpy()
        labels = torch.cat([o['label'] for o in outputs]).cpu().numpy()
        half_idx = len(ys) // 2
        classifier = LogisticRegression(max_iter=100, solver="liblinear")
        classifier.fit(ys[:half_idx], labels[:half_idx])
        val_acc = 100 * classifier.score(ys[half_idx:], labels[half_idx:])
        logs = {'val_acc': torch.tensor(val_acc)}
        results = {'log': logs}
        return results

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        optimizer = LARSWrapper(optimizer)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]

    @classmethod 
    def extract_kwargs_from_argparse_args(cls, args, **kwargs):
        return extract_kwargs_from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1.5e-6)
        parser.add_argument('--warmup_epochs', type=int, default=10)
        return parser