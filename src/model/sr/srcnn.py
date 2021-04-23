from model import common

from torch import nn

class SRCNN(nn.Sequential):

    def __init__(self, n_colors=3):
        m = [
            nn.Conv2d(n_colors, 64, 9, padding=4),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 5, padding=2),
        ]
        super().__init__(*m)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        kwargs = {
            'n_colors': cfg.n_colors,
        }
        return kwargs

