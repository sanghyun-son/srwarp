import typing

import misc
import torch
import torch.nn as nn


class PSNR(nn.Module):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        if 'y' in name:
            coeffs = [65.738, 129.057, 25.064]
            coeffs = torch.Tensor(coeffs).view(1, 3, 1, 1) / 256
            self.register_buffer('coeffs', coeffs)
        else:
            self.coeffs = None

        self.shave = 8
        return

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            shave: typing.Optional[int]=None) -> torch.Tensor:

        # Unable to calculate the PSNR
        if x.size() != y.size():
            return 0

        if shave is None:
            shave = self.shave

        # Since we assume -1 ~ 1 range
        diff = 0.5 * (x - y)
        if self.coeffs is not None and diff.size(1) > 1:
            diff = diff.mul(self.coeffs).sum(dim=1)

        side = min(x.size(-2), x.size(-1))
        if shave > 0 and side >= 64:
            # Do not shave the patches if they are small
            valid = diff[..., shave:-shave, shave:-shave]
        else:
            valid = diff

        mse = valid.pow(2).mean()
        psnr = -10 * mse.log10()
        return psnr

