import misc
import torch
import torch.nn as nn

from loss.measure_fn import psnr


class MaskedPSNR(psnr.PSNR):

    def __init__(self, name):
        super().__init__(name)

    def forward(self, x, y, mask):
        # Unable to calculate the PSNR
        if x.size() != y.size():
            return 0

        # Since we assume -1 ~ 1 range
        diff = 0.5 * mask * (x - y)
        if self.coeffs is not None and diff.size(1) > 1:
            diff = diff.mul(self.coeffs).sum(dim=1)

        side = min(x.size(-2), x.size(-1))
        if self.shave > 0 and side >= 64:
            # Do not shave the patches if they are small
            valid = diff[..., self.shave:-self.shave, self.shave:-self.shave]
        else:
            valid = diff

        gain = mask.nelement() / mask.sum()
        mse = gain.item() * valid.pow(2).mean()
        psnr = -10 * mse.log10()
        return psnr

