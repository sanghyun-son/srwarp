import math
import typing

import torch

_TT = typing.Tuple[torch.Tensor, torch.Tensor]

@torch.no_grad()
def functional_grid(
        sizes: typing.Tuple[int, int],
        f: typing.Callable[[_TT], torch.Tensor],
        scale: float=1,
        eps_y: float=0,
        eps_x: float=0) -> torch.Tensor:

    n = sizes[0] * sizes[1]
    r = torch.arange(n)
    r = r.float()
    r = r.cuda()
    x = r % sizes[1] + eps_x
    y = r // sizes[1] + eps_y
    grid_source = f(x, y)
    grid_source = torch.stack(grid_source, dim=0)
    scale_inv = 1 / scale
    grid_source *= scale_inv
    grid_source += 0.5 * (scale_inv - 1)
    return grid_source

@torch.no_grad()
def f_sine(
        xp: torch.Tensor,
        yp: torch.Tensor,
        a: float=2,
        t: float=12,
        scale: float=2,
        backward: bool=True) -> _TT:

    if backward:
        x = xp / scale + 0.5 * (1 / scale - 1)
        y = yp / scale - a * torch.sin(x / t) - a + 0.5 * (1 / scale - 1)
    else:
        x = xp
        y = yp + a * torch.sin(xp / t)

    return x, y

@torch.no_grad()
def f_barrel(
        xp: torch.Tensor,
        yp: torch.Tensor,
        hp: int=512,
        wp: int=512,
        k: float=1,
        scale: float=2,
        backward: bool=True) -> _TT:

    if backward:
        hpc = (hp - 1) / 2
        wpc = (wp - 1) / 2
        xd = (xp - wpc) / wp
        yd = (yp - hpc) / hp
        rp_pow = xd.pow(2) + yd.pow(2)
        rp = rp_pow.sqrt()
        r = rp * (1 + k * rp_pow)
        factor = r / (scale * rp + 1e-6)
        hc = (hp / 2 - 1) / scale
        wc = (wp / 2 - 1) / scale
        x = factor * (xp - wpc) + wc + 0.5 * (1 / scale - 1)
        y = factor * (yp - hpc) + hc + 0.5 * (1 / scale - 1)

    return x, y

@torch.no_grad()
def f_spiral(
        xp: torch.Tensor,
        yp: torch.Tensor,
        hp: int=512,
        wp: int=512,
        k: float=1,
        scale: float=2,
        backward: bool=True) -> _TT:

    if backward:
        s = math.sqrt(hp**2 + wp**2)
        xo = (s - wp) / 2
        yo = (s - hp) / 2

        '''
        hpc = (hp - 1) / 2
        wpc = (wp - 1) / 2
        xd = (xp - wpc) / wp
        yd = (yp - hpc) / hp
        rp = xd.pow(2) + yd.pow(2)
        cos = torch.cos(k * rp)
        sin = torch.sin(k * rp)
        xt = (xp - wpc) * cos + (yp - hpc) * sin + wpc
        yt = -(xp - wpc) * sin + (yp - hpc) * cos + hpc
        x = xt / scale + 0.5 * (1 / scale - 1)
        y = yt / scale + 0.5 * (1 / scale - 1)
        '''
        sc = (s - 1) / 2
        xd = (xp - sc) / wp
        yd = (yp - sc) / hp
        rp = xd.pow(2) + yd.pow(2)
        cos = torch.cos(k * rp)
        sin = torch.sin(k * rp)
        xt = (xp - sc) * cos + (yp - sc) * sin + sc
        yt = -(xp - sc) * sin + (yp - sc) * cos + sc
        x = xt / scale + 0.5 * (1 / scale - 1) - xo / scale
        y = yt / scale + 0.5 * (1 / scale - 1) - yo / scale

    return x, y

@torch.no_grad()
def f_lambda(
        f: typing.Callable,
        *args,
        **kwargs) -> typing.Callable[[_TT], torch.Tensor]:

    l = lambda x, y: f(x, y, *args, **kwargs)
    return l

