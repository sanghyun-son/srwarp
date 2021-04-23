import numpy as np

import torch
from torch import nn


class PartialConv(nn.Conv2d):

    def __init__(self, *args, **kwargs) -> None:
        kwargs['padding_mode'] = 'reflect'
        super().__init__(*args, bias=True, **kwargs)
        for m in self.modules():
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0)

        self.partial_conv = nn.Conv2d(
            1,
            1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False,
            padding_mode='reflect',
        )
        self.partial_conv.weight.data.fill_(1 / np.prod(self.kernel_size))
        self.partial_conv.requires_grad_(False)
        return

    def _forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): (B, C, H, W)
            mask (torch.Tensor): (1, 1, H, W)

        Return:
        '''
        x = mask * x
        with torch.no_grad():
            w = self.partial_conv(mask)
            w.clamp_(min=1e-8)
            w.reciprocal_()
            w *= mask

        x = super().forward(x)
        x *= w
        return x

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            #padding_mode: str='reflect') -> torch.Tensor:
            padding_mode: str='zero') -> torch.Tensor:

        x = x * mask
        #print(padding_mode)
        if padding_mode == 'zero':
            x = super().forward(x)
            x = x * mask
            return x

        with torch.no_grad():
            b, c, h, w = x.size()
            x_pad = x.view(b * c, 1, h, w)
            x_pad = self.partial_conv(x_pad)
            x_pad = x_pad.view(b, c, h, w)

            weight = self.partial_conv(mask)
            weight.clamp_(min=1e-4)
            weight.reciprocal_()
            weight_after = weight * mask

            void = (weight > np.prod(self.kernel_size) + 1).float()
            weight *= (1 - void) * (1 - mask)

        x = x + x_pad * weight
        x = super().forward(x)
        x = x * weight_after
        return x
