import random
import types
import typing

import torch


class Preprocessing(object):

    def __init__(self, augmentation: str='hvr') -> None:
        self.augmentation = augmentation
        return

    def _apply(self, f: typing.Callable, **kwargs) -> dict:
        applied = {k: f(v) for k, v in kwargs.items()}
        return applied

    @torch.no_grad()
    def augment(self, augmentation: typing.Optional[str], **kwargs) -> dict:
        if augmentation is None:
            augmentation = self.augmentation

        hflip = 'h' in augmentation and random.random() < 0.5
        vflip = 'v' in augmentation and random.random() < 0.5
        rot90 = 'r' in augmentation and random.random() < 0.5

        def _augment(x: torch.Tensor) -> torch.Tensor:
            if hflip:
                x = x.flip(-1)

            if vflip:
                x = x.flip(-2)

            if rot90:
                x = x.transpose(-2, -1)

        applied = self._apply(_augment, **kwargs)
        return applied
