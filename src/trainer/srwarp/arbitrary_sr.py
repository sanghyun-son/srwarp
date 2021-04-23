import math
import random
import types

from trainer import base_trainer
from misc import image_utils

import torch
from torch import cuda

from srwarp import transform
from srwarp import resize
from srwarp import wtypes

_parent_class = base_trainer.BaseTrainer


class SuperWarpTrainer(_parent_class):

    def __init__(
            self,
            *args,
            scale: float=4,
            scale_min: float=1.1,
            scale_max: float=4,
            reset_kernel: bool=False,
            reset_sampler: bool=False,
            **kwargs) -> None:

        self.reset_kernel = reset_kernel
        self.reset_sampler = reset_sampler
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.scale_min = scale_min
        self.scale_max = scale_max
        if scale_min == 1.1:
            self.scales = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
            ]
        else:
            self.scales = [
                2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
            ]

        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['scale'] = cfg.scale
        kwargs['scale_min'] = cfg.scale_min
        kwargs['scale_max'] = cfg.scale_max
        kwargs['reset_kernel'] = cfg.reset_kernel
        kwargs['reset_sampler'] = cfg.reset_sampler
        return kwargs

    def preprocess_state(self, state: dict) -> dict:
        pop_list = []
        for k in state.keys():
            name = k.split('.')[0]
            if self.reset_kernel and name == 'k':
                pop_list.append(k)

        for p in pop_list:
            state.pop(p)

        return state

    @torch.no_grad()
    def get_input(self, hr: torch.Tensor, s: float) -> torch.Tensor:
        lr = 127.5 * (hr + 1)
        lr = resize.imresize(
            lr,
            scale=(1 / s),
            kernel_type='bicubic',
            range_8bit=True,
        )
        lr = lr / 127.5 - 1
        return lr

    def forward(self, **samples) -> wtypes._TT:
        if 'img' in samples:
            hr = samples['img']
            if self.training:
                s = random.choice(self.scales)
                self.rt_log_postfix = '(x{:.1f})'.format(s)
            else:
                s = self.scale

            lr = self.get_input(hr, s)
            m = transform.scaling(s)
            # m would not be changed.
            # Only for calculating the output dimension.
            m, sizes, _ = transform.compensate_matrix(lr, m)
        else:
            lr = samples['lr']
            hr = samples['hr']
            m = transform.scaling(self.scale)
            _, _, h, w = lr.size()
            sizes = (math.ceil(h * self.scale), math.ceil(w * self.scale))

        m = transform.replicate_matrix(m, do_replicate=self.training)

        sr, _ = self.pforward(lr, m, sizes=sizes)
        if not self.training:
            sr = image_utils.quantize(sr)

        # To make it compatible with the Meta-SR evaluation
        if 'img' not in samples:
            sizes = (int(h * self.scale), int(w * self.scale))

        min_h = min(sizes[0], hr.size(-2))
        min_w = min(sizes[1], hr.size(-1))
        sr = sr[..., :min_h, :min_w]
        hr = hr[..., :min_h, :min_w]
        #self.pause(count_max=10, sr=sr, lr=lr, hr=hr)
        loss = self.loss(sr=sr, hr=hr)
        return loss, sr
