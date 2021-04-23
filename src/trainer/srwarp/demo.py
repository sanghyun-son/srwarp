import typing

from trainer import base_trainer
from misc import image_utils

import torch

from srwarp import wtypes

_parent_class = base_trainer.BaseTrainer


class SRWarpDemo(_parent_class):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return

    def forward(self, **samples) -> wtypes._TT:
        x = samples['img']
        gt = samples['gt']
        m = samples['m'][0]

        y, mask = self.pforward(x, m, sizes=(gt.size(-2), gt.size(-1)))
        y = image_utils.quantize(y)
        loss = self.loss(
            sr=y,
            hr=gt,
            mask=mask,
        )

        return loss, y
