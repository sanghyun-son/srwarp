from model import common

import torch
from torch import nn


class EDSRF(nn.Module):
    '''
    EDSRF model for extracting high-resolution feature.

    Note:
        From Lim et al.,
        "Enhanced Deep Residual Networks for Single Image Super-Resolution"
        See https://arxiv.org/pdf/1707.02921.pdf for more detail.
    '''

    def __init__(
            self,
            scale: int=4,
            depth: int=16,
            n_colors: int=3,
            n_feats: int=64) -> None:

        super().__init__()
        self.n_colors = n_colors
        self.conv = common.default_conv(n_colors, n_feats, 3)
        resblock = lambda: common.ResBlock(n_feats, 3)
        m = [resblock() for _ in range(depth)]
        m.append(common.default_conv(n_feats, n_feats, 3))
        self.resblocks = nn.Sequential(*m)
        self.recon = nn.Sequential(common.Upsampler(scale, n_feats))
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: torch.Tensor: (B, C, H, W)

        Return:
            torch.Tensor: (B, C, sH, sW)
        '''
        x = self.conv(x)
        x = x + self.resblocks(x)
        x = self.recon(x)
        return x 
