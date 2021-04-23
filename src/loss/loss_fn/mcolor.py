import torch
from torch import nn
from torch.nn import functional as F


class MaskedColor(nn.Module):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.__name = name
        return

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:

        hy = y.size(-2)
        wy = y.size(-1)
        hm = mask.size(-2)
        wm = mask.size(-1)

        if (hy != hm) and (wy != wm):
            print('x:', x.size())
            print('y:', y.size())
            print('mask:', mask.size())
            raise ValueError('y should have the same dimension to mask!')

        # (B, C, H, W) each
        masked_y = mask * y
        # (B, C) each
        mean_x = x.mean((-2, -1))
        mean_y = masked_y.mean((-2, -1))
        gain_y = mask.nelement() / mask.sum()
        loss = F.l1_loss(mean_x, gain_y * mean_y)
        return loss
