from misc.gpu_utils import parallel_forward as pforward

import torch
from torch import nn
from torch.nn import parallel


def x8_forward(model, *args, rep=1, **kwargs):
    tfs = ('v', 'h', 't')
    def gtransform(x, tf):
        if tf == 'v':
            return x.flip(-1)
        elif tf == 'h':
            return x.flip(-2)
        elif tf == 't':
            return x.transpose(-1, -2)

    def augment(x_augs, tf):
        n = len(x_augs)
        for x_aug in x_augs[:n]:
            x_augs.append([gtransform(x, tf) for x in x_aug])

        return x_augs

    def merge(x_augs, tf):
        y = []
        half = len(x_augs) // 2
        inv = lambda x: gtransform(x, tf)
        for x_augf, x_augb in zip(x_augs[:half], x_augs[half:]):
            if isinstance(x_augf, (list, tuple)):
                y.append([(xf + inv(xb)) / 2 for xf, xb in zip(x_augf, x_augb)])
            else:
                y.append((x_augf + inv(x_augb)) / 2)

        return y

    def forward(*x):
        if rep == 1:
            return pforward(model, x, **kwargs)
        else:
            x = x[0]
            for _ in range(rep):
                x = pforward(model, x, **kwargs)

        return x

    '''
    x0, x1, ...: Input tensors
    y0, y1, ...: Output tensors
    m_y0, m_y1, ...: Merged output tensors
    f, g, ... : Geometric transformation
    '''
    # [[x0, x1, ...]]
    x_augs = [args]
    for tf in tfs:
        # [[x0, x1, ...], [f(x0), f(x1), ...], [g(x0), g(x1), ...], ...]
        x_augs = augment(x_augs, tf)

    #[[y0, y1, ...], [y(f(x0)), y(f(x1)), ...], [y(g(x0)), y(g(x1)), ...]]]
    y_augs = [forward(*x_aug) for x_aug in x_augs]
    for tf in reversed(tfs):
        y_augs = merge(y_augs, tf)

    # [[m_y0, m_y1, ...]]
    y = y_augs[0]
    return y


def quad_forward(model, *args, **kwargs):
    '''
    Args:
        args (list): List of input tensors [x1, x2, ...].

    Return:
    '''
    max_size = 256**2
    boundary = 8

    def divide(x, state, depth=0):
        h = x.size(-2)
        w = x.size(-1)
        state.append((h, w))
        if h * w <= max_size:
            return x

        top = slice(None, h//2 + boundary)
        bottom = slice(h - h//2 - boundary, h)
        left = slice(None, w//2 + boundary)
        right = slice(w - w//2 - boundary, w)
        tl = x[..., top, left]
        tr = x[..., top, right]
        bl = x[..., bottom, left]
        br = x[..., bottom, right]
        quads = torch.cat([tl, tr, bl, br], dim=0)
        print(quads.size())
        # Recursive cut-down.
        quads = divide(quads, state, depth=depth + 1)
        return quads

    xs = []
    states = []
    for arg in args:
        state = []
        quads = divide(arg, state)
        xs.append(quads)
        states.append(state)

    # xs: [[x11, x12, ...], [x21, x22, ...], ...]
    gpus = torch.cuda.device_count()
    # xs: [[[x11, x12], [x13, x14], ...], [[x21, x22], [x23, x24], ...], ...]
    xs = [x.split(gpus, dim=0) for x in xs]
    ys = []
    for s in zip(*xs):
        x = pforward(model, *s, **kwargs)
        if len(args) == 1:
            x = (x,)

        # [[y11, y21, ...], [y12, y22, ...], ...]
        ys.append(x)

    def conquer(x, state):
        if len(state) == 1:
            return x

        _, w = state[-1]
        hp, wp = state[-2]
        scale = x.size(-1) / w
        mod_h = int(scale * (hp % 2))
        mod_w = int(scale * (wp % 2))
        boundary_s = int(scale * boundary)
        tl, tr, bl, br = x.chunk(4, dim=0)
        tl = tl[..., :-boundary_s + mod_h, :-boundary_s + mod_w]
        tr = tr[..., :-boundary_s + mod_h, boundary_s:]
        bl = bl[..., boundary_s:, :-boundary_s + mod_w]
        br = br[..., boundary_s:, boundary_s:]
        top = torch.cat((tl, tr), dim=-1)
        bottom = torch.cat((bl, br), dim=-1)
        quads = torch.cat((top, bottom), dim=-2)
        quads = conquer(quads, state[:-1])
        return quads

    # [cat(y11, y12, ...), cat(y21, y22, ...), ...]
    ys = [torch.cat(y, dim=0) for y in zip(*ys)]
    y_final = []
    for y, state in zip(ys, states):
        # [y1, y2, ...]
        y_final.append(conquer(y, state))

    if len(args) == 1:
        y_final = y_final[0]

    return y_final
