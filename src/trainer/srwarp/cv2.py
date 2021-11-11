import time
import types
import typing

from trainer.srwarp import warptrainer
from misc import image_utils

import numpy as np
import torch
import cv2

from srwarp import transform
from srwarp import warp
from srwarp import wtypes

_parent_class = warptrainer.SRWarpTrainer


class CV2Predictor(_parent_class):

    def __init__(
            self,
            *args,
            scale: int=4,
            naive: bool=False,
            interpolation: str='bicubic',
            is_demo: bool=False,
            **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.model.fill = -255
        self.scale = scale
        self.naive = naive
        self.interpolation = interpolation
        self.is_demo = is_demo
        self.time_acc = 0
        self.count = 0
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['scale'] = cfg.scale
        kwargs['naive'] = cfg.cv2_naive
        kwargs['interpolation'] = cfg.cv2_interpolation
        kwargs['is_demo'] = ('srwarp.demo' in cfg.dtest)
        return kwargs

    def forward(self, **samples) -> typing.Tuple[torch.Tensor, float]:
        if self.is_demo:
            hr = samples['gt']
            lr_crop = samples['img']
            m = samples['m'][0].cpu()
        else:
            hr = samples['img']
            m_inv = samples['m'][0].cpu()
            lr_crop, m = self.get_input(hr, m_inv)

        sizes = (hr.size(-1), hr.size(-2))

        if self.naive:
            sr = lr_crop
        else:
            sr = self.model(lr_crop)

        m = transform.compensate_scale(m, 1 / self.scale)
        if self.interpolation == 'bicubic':
            flags = cv2.INTER_CUBIC
        elif self.interpolation == 'area':
            flags = cv2.INTER_AREA
        elif self.interpolation == 'lanczos':
            flags = cv2.INTER_LANCZOS4
        elif self.interpolation == 'layer':
            flags = 'layer'
        elif self.interpolation == 'adaptive':
            flags = 'adaptive'

        fill = -255
        if self.interpolation in ('bicubic', 'area', 'lanczos'):
            sr_np = sr.squeeze(0).cpu().numpy()
            sr_np = np.transpose(sr_np, (1, 2, 0))
            sr_np = np.ascontiguousarray(sr_np)
            m_np = m.cpu().numpy()
            sr_cv2_np = cv2.warpPerspective(
                sr_np,
                m_np,
                sizes,
                flags=flags,
                borderMode=cv2.BORDER_REFLECT,
            )
            sr_cv2_np = np.transpose(sr_cv2_np, (2, 0, 1))
            sr_cv2_np = np.ascontiguousarray(sr_cv2_np)

            ref = warp.warp_by_function(
                sr,
                m,
                f_inverse=False,
                sizes=(sizes[1], sizes[0]),
                fill=fill,
            )
            mask = (ref != fill).float()
            sr_cv2 = torch.from_numpy(sr_cv2_np).to(sr.device).unsqueeze(0)
        else:
            adaptive_grid = (flags == 'adaptive')
            sr_cv2 = warp.warp_by_function(
                sr,
                m,
                f_inverse=False,
                sizes=(sizes[1], sizes[0]),
                adaptive_grid=adaptive_grid,
                fill=fill,
            )
            mask = (sr_cv2 != fill).float()

        sr_cv2 = image_utils.quantize(sr_cv2)
        sr_cv2 = mask * (sr_cv2 + 1) - 1
        loss = self.loss(sr=sr_cv2, hr=hr, mask=mask)
        return loss, {'sr': sr_cv2, 'mask': 2 * mask - 1}
