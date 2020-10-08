import torch
import torch.nn as nn

from byol import BYOL, default_augmentation
from stl10_datamodule import STL10DataModule
from train_utils import main


class BYOLSTL10(BYOL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = default_augmentation(96, is_color_image=True)
        mobilenet = torch.hub.load(
            'pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)
        mobilenet.classifier = nn.Identity()
        self.online_encoder = mobilenet
        self.copy()


byol_args = {
    "projector_isize": 1280,
    "projector_hsize": 2560,
    "projector_osize": 256,
    "predictor_hsize": 512
}


if __name__ == '__main__':
    main(STL10DataModule, BYOLSTL10, byol_args, "byol_stl10")
