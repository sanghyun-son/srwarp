import torch
from torch import nn

from model.srwarp import pconv


class PartialResBlock(nn.Module):

    def __init__(self, n_feats: int=64, kernel_size: int=3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = pconv.PartialConv(
            n_feats,
            n_feats,
            kernel_size,
            padding=padding,
            padding_mode='reflect',
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = pconv.PartialConv(
            n_feats,
            n_feats,
            kernel_size,
            padding=padding,
            padding_mode='reflect',
        )
        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = mask * x
        res = self.conv1(x, mask)
        res = self.relu(res)
        res = self.conv2(res, mask)
        x = x + res
        return x


class PartialResSeq(nn.ModuleList):

    def __init__(
            self,
            n_inputs: int=64,
            n_feats: int=64,
            n_outputs: int=3,
            kernel_size: int=3,
            depth: int=4) -> None:

        super().__init__()

        args = [kernel_size]
        kwargs = {
            'padding': kernel_size // 2,
            'padding_mode': 'reflect',
        }
        if n_inputs == n_feats:
            self.append(PartialResBlock(
                n_feats=n_feats, kernel_size=kernel_size,
            ))
            depth -= 2
        else:
            self.append(pconv.PartialConv(n_inputs, n_feats, *args, **kwargs))
            depth -= 1

        while depth > 0:
            self.append(PartialResBlock(
                n_feats=n_feats, kernel_size=kernel_size,
            ))
            depth -= 2

        if n_outputs != n_feats:
            self.append(pconv.PartialConv(n_feats, n_outputs, *args, **kwargs))

        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for m in self:
            x = m(x, mask)

        return x


class PartialResRecon(nn.Module):

    def __init__(
            self,
            n_inputs: int=64,
            n_feats: int=64,
            n_outputs: int=3,
            kernel_size: int=3,
            depth: int=4) -> None:

        super().__init__()
        self.body = PartialResSeq(
            n_inputs=n_feats,
            n_feats=n_feats,
            n_outputs=n_feats,
            kernel_size=kernel_size,
            depth=depth,
        )
        self.recon = pconv.PartialConv(
            n_feats,
            n_outputs,
            kernel_size,
            padding=(kernel_size // 2),
            padding_mode='reflect',
        )
        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.body(x, mask)
        x = self.recon(x, mask)
        return x
