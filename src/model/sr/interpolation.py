import torch
from torch import nn
from torch.nn import functional

def model_class(cfg, conv=None, make=False):
    if make:
        return Interpolation(**Interpolation.get_kwargs(cfg=cfg))
    else:
        return Interpolation


class Interpolation(nn.Module):

    def __init__(self, scale=4, mode='bilinear'):
        super(Interpolation, self).__init__()
        self.scale = scale
        self.mode = mode
        self.dummy = nn.Parameter(torch.zeros(0))

    @staticmethod
    def get_kwargs(cfg):
        return {
            'scale': cfg.scale,
        }

    def forward(self, x):
        x = functional.interpolate(
            x,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=True if self.mode == 'bilinear' else None,
        )
        return x
