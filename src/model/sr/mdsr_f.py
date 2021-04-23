import typing

from model import common

import torch
from torch import nn


class MDSRF(nn.Module):
    '''
    EDSRF model for extracting high-resolution features.

    Note:
        From Lim et al.,
        "Enhanced Deep Residual Networks for Single Image Super-Resolution"
        See https://arxiv.org/pdf/1707.02921.pdf for more detail.
    '''

    def __init__(
            self,
            depth: int=16,
            n_colors: int=3,
            n_feats: int=64,
            multi_scale: bool=True,
            bottleneck: typing.Optional[int]=None) -> None:

        super().__init__()
        kwargs = {'padding_mode': 'reflect'}
        #kwargs = {}
        self.conv = common.default_conv(n_colors, n_feats, 3, **kwargs)
        resblock = lambda: common.ResBlock(n_feats, 3, **kwargs)
        m = [resblock() for _ in range(depth)]
        m.append(common.default_conv(n_feats, n_feats, 3, **kwargs))
        self.resblocks = nn.Sequential(*m)

        self.recon_x1 = common.default_conv(n_feats, n_feats, 3, **kwargs)
        if multi_scale:
            self.recon_x2 = common.Upsampler(2, n_feats, **kwargs)
            self.recon_x4 = common.Upsampler(4, n_feats, **kwargs)

        self.multi_scale = multi_scale
        if bottleneck is None:
            self.bottleneck_x1 = None
            self.bottleneck_x2 = None
            self.bottleneck_x4 = None
        else:
            self.bottleneck_x1 = common.default_conv(n_feats, bottleneck, 1)
            self.bottleneck_x2 = common.default_conv(n_feats, bottleneck, 1)
            self.bottleneck_x4 = common.default_conv(n_feats, bottleneck, 1)

        #print(self)
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
        x1 = self.recon_x1(x)
        if self.bottleneck_x1 is not None:
            x1 = self.bottleneck_x1(x1)

        if self.multi_scale:
            x2 = self.recon_x2(x)
            if self.bottleneck_x2 is not None:
                x2 = self.bottleneck_x2(x2)

            x4 = self.recon_x4(x)
            if self.bottleneck_x4 is not None:
                x4 = self.bottleneck_x4(x4)

            return x1, x2, x4
        else:
            return x1

