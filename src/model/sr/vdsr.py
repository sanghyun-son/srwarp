from bicubic_pytorch import core
from config import get_config
from model import common

import torch
from torch import nn
from torch.nn import init


class VDSR(nn.Module):

    def __init__(
            self,
            depth: int=20,
            n_colors: int=3,
            n_feats: int=64,
            conv=common.default_conv) -> None:

        super().__init__()
        m = []
        block = lambda x, y: common.BasicBlock(x, y, 3)
        m.append(block(n_colors, n_feats))
        for _ in range(depth - 2):
            m.append(block(n_feats, n_feats))

        m.append(conv(n_feats, n_colors, 3))
        self.convs = nn.Sequential(*m)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0)

        return

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv) -> dict:
        parse_list = ['depth', 'n_colors', 'n_feats']
        kwargs = get_config.parse_namespace(cfg, *parse_list)
        kwargs['conv'] = conv
        return kwargs

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        with torch.no_grad():
            x = core.imresize(x, scale=scale, kernel='cubic')

        x = x + self.convs(x)
        return x 

