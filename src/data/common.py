import io
import random

import imageio
import numpy as np
from skimage import color

import torch
from torch.nn import functional as F

def resize_mask(mask, scale):
    '''
    Args:
        mask
        scale
    '''
    with torch.no_grad():
        is_bool = isinstance(mask, torch.BoolTensor)
        is_cbool = isinstance(mask, torch.cuda.BoolTensor)
        # F.interpolate doesn't work with BoolTensor
        if is_bool or is_cbool:
            mask = mask.float()

        offset = scale // 2
        s = 1 / scale
        mask_new = torch.zeros_like(mask)
        mask_new[..., :-offset, :-offset] = mask[..., offset:, offset:]
        if mask_new.ndim == 3:
            mask_new.unsqueeze_(0)

        mask_new = F.interpolate(mask_new, scale_factor=s, mode='nearest')

        # Back to Bool
        if is_bool or is_cbool:
            mask_new = mask_new.bool()

    return mask_new

def make_pre(cfg):
    return Preprocessing(
        scale=cfg.scale,
        patch=cfg.patch,
        n_colors=cfg.n_colors,
        augmentation=cfg.augmentation,
        noise=cfg.noise,
        compression=cfg.compression,
    )


class Preprocessing(object):

    def __init__(
            self, scale=2, patch=96, n_colors=3,
            augmentation='hvr', noise=None, compression=None):
        
        self.scale = scale
        self.patch = patch
        self.n_colors = n_colors
        self.augmentation = augmentation
        self.noise = noise
        self.compression = compression
        self.multi_scale = False

    def apply(self, func, ignore=None, **kwargs):
        if ignore is None:
            ignore = ()
        elif not isinstance(ignore, (tuple, list)):
            ignore = (ignore,)

        include = {k: v for k, v in kwargs.items() if k not in ignore}
        exclude = {k: v for k, v in kwargs.items() if k in ignore}
        filter_tensor = lambda x: x if isinstance(x, torch.Tensor) else func(x)
        applied = {k: filter_tensor(v) for k, v in include.items()}
        return {**applied, **exclude}

    def modcrop(self, scale=None, **kwargs):
        if scale is None:
            scale = self.scale

        if not self.scale.is_integer():
            return kwargs

        h = min(v.shape[0] for v in kwargs.values())
        w = min(v.shape[1] for v in kwargs.values())
        th = int(scale * h)
        tw = int(scale * w)
        def _modcrop(img):
            return img[0:th, 0:tw]

        return self.apply(_modcrop, **kwargs)

    def get_patch(self, patch=None, **kwargs):
        if patch is None:
            patch = self.patch

        h = min(v.shape[0] for v in kwargs.values())
        w = min(v.shape[1] for v in kwargs.values())
        iy = random.randrange(0, h - patch + 1)
        ix = random.randrange(0, w - patch + 1)
        def _get_patch(img):
            scale = img.shape[0] // h
            tx = scale * ix
            ty = scale * iy
            tp = scale * patch
            return img[ty:ty + tp, tx:tx + tp]

        return self.apply(_get_patch, **kwargs)

    def set_color(self, n_colors=None, **kwargs):
        if n_colors is None:
            n_colors = self.n_colors

        def _set_color(img):
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            c = img.shape[2]
            # Do not process multi-channel tensor
            if c > 3 or n_colors == c:
                return img

            if n_colors == 1:
                y = color.rgb2ycbcr(img)[..., 0]
                img = np.expand_dims(y, 2)
            elif c == 1:
                img = img.repeat(n_colors, axis=-1)

            return img

        return self.apply(_set_color, **kwargs)

    def np2Tensor(self, **kwargs):
        '''
        Convert numpy array into tensor
        Args:

        Return:

        Note:
            Input range is 0 ~ 255
            Output range is -1 ~ 1
        '''
        def _np2Tensor(img):
            is_bool = (img.dtype == np.bool)

            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            if not is_bool:
                img = img.float()
                img = (img / 127.5) - 1

            return img

        return self.apply(_np2Tensor, **kwargs)

    def augment(self, augmentation=None, **kwargs):
        if augmentation is None:
            augmentation = self.augmentation

        if not augmentation:
            return kwargs

        hflip = 'h' in augmentation and random.random() < 0.5
        vflip = 'v' in augmentation and random.random() < 0.5
        rot90 = 'r' in augmentation and random.random() < 0.5
        if 's' in augmentation:
            order = [i for i in range(self.n_colors)]
            random.shuffle(order)
        else:
            order = None

        if 'n' in augmentation:
            negative = [random.random() < 0.5 for _ in range(self.n_colors)]
        else:
            negative = None

        def _augment(img):
            if hflip:
                img = img[:, ::-1]
            if vflip:
                img = img[::-1, :]
            if rot90:
                img = np.moveaxis(img, 1, 0)
            if order is not None:
                img = img[..., order]
            if negative is not None:
                for i, neg in enumerate(negative):
                    if neg:
                        img[..., i] = 255 - img[..., i]
            
            return img

        return self.apply(_augment, **kwargs)

    def add_noise(self, noise=None, **kwargs):
        if noise is None:
            noise = self.noise

        if not noise:
            return kwargs

        split = noise.split('-')
        if len(split) == 1:
            sigma = int(split[0])
        else:
            # Sample sigma from a given range
            lb, ub = [int(s) for s in split]
            sigma = random.randrange(lb, ub + 1)

        def _add_noise(img):
            n = sigma * np.random.randn(*img.shape)
            img_n = img.astype(np.float32) + n
            img_n = img_n.round().clip(min=0, max=255)
            img_n = img_n.astype(np.uint8)
            return img_n

        return self.apply(_add_noise, ignore='hr', **kwargs)

    def compress(self, compression=None, **kwargs):
        if compression is None:
            compression = self.compression

        if not compression:
            return kwargs

        if len(compression) == 1 and '-' in compression[0]:
            lb, ub = [int(x) for x in compression[0].split('-')]
            quality = random.randrange(lb, ub + 1)
            length = (ub - lb) / 100
            jpg = random.random() > max(0.1, 0.5 - length)
        else:
            quality = int(random.choice(compression))
            jpg = random.random() > 1 / (1 + len(compression))

        if not jpg:
            return kwargs

        def to_jpg(k, img):
            if k != 'hr':
                with io.BytesIO() as f:
                    imageio.imwrite(f, img, format='jpg', quality=quality)
                    f.seek(0)
                    img = imageio.imread(f)

            return img

        kwargs = {k: to_jpg(k, v) for k, v in kwargs.items()}
        return kwargs

