from data import common
from data.sr import dataclass
import numpy as np
from PIL import Image

_parent_class = dataclass.SRData


class Demo(_parent_class):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_kwargs(cfg, train=False):
        kwargs = _parent_class.get_kwargs(cfg, train=False)
        kwargs['is_binary'] = False
        return kwargs

    def get_path(self, degradation, scale):
        path_lr = self.dpath
        path_dict = {'lr': path_lr, 'hr': path_lr}
        return path_dict

    def get_patch(self, **kwargs):
        pil = Image.fromarray(kwargs['hr'])
        w, h = pil.size
        w = int(self.scale[0] * w)
        h = int(self.scale[0] * h)
        pil = pil.resize((w, h), resample=Image.BICUBIC)
        arr = np.array(pil)
        img_dict = {'lr': kwargs['lr'], 'hr': arr}
        return img_dict

