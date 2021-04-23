import os
from os import path
import glob
import abc
import pickle

from config import get_config
from data import common

import numpy as np
import imageio

import torch
from torch import utils


class SRBase(utils.data.Dataset, metaclass=abc.ABCMeta):

    def __init__(
            self,
            dpath: str,
            scale: float=4,
            degradation: str='bicubic',
            preprocessing=None,
            train: bool=True):

        super().__init__()
        self.dpath = dpath
        self.scale = scale
        self.degradation = degradation
        self.pre = preprocessing
        self.train = train
        self.data = self.scan()

        if 'hr' not in self.data:
            raise KeyError('HR images do not exist!')

    def __len__(self) -> int:
        return len(self.data['hr'])

    def __getitem__(self, idx: int) -> dict:
        name = self.data['hr'][idx]
        name = path.splitext(path.basename(name))[0]
        name_dict = {k: v[idx] for k, v in self.data.items()}
        img_dict = {k: self.load_file(v) for k, v in name_dict.items()}
        img_dict = self.pre.modcrop(**img_dict)
        if self.train:
            img_dict = self.pre.get_patch(**img_dict)
            img_dict = self.pre.augment(**img_dict)

        img_dict = self.pre.set_color(**img_dict)
        img_dict = self.pre.np2Tensor(**img_dict)
        img_dict['name'] = name
        return img_dict

    @staticmethod
    def get_kwargs(cfg, train: bool=True) -> dict:
        parse_list = [
            'dpath',
            'scale',
            'degradation'
        ]
        kwargs = get_config.parse_namespace(cfg, *parse_list)
        kwargs['preprocessing'] = common.make_pre(cfg)
        kwargs['train'] = train
        return kwargs

    def cache(self, k: str) -> np.array:
        cache_ext = ('.png',)
        ext = path.splitext(k)[-1]
        if ext in cache_ext:
            cache_dir = path.join(self.dpath, 'cache')
            cache_name = k.replace(self.dpath, cache_dir)
            cache_name = cache_name.replace(ext, '.npy')
            if not path.isfile(cache_name):
                v = imageio.imread(k)
                h = v.shape[0]
                w = v.shape[1]
                # Minumum size
                cache_res = 512 * 512
                if h * w >= cache_res:
                    os.makedirs(path.dirname(cache_name), exist_ok=True)
                    with open(cache_name, 'wb') as f:
                        pickle.dump(v, f)
            else:
                with open(cache_name, 'rb') as f:
                    v = pickle.load(f)
        else:
            v = imageio.imread(k)

        return v

    def get_path(self) -> str:
        raise NotImplementedError

    def scan_dirs(self) -> dict:
        '''
        Specify LR and HR image directories here.
        '''
        dpath = self.get_path()
        path_hr = path.join(dpath, 'HR')
        path_lr = path.join(dpath, 'LR_{}'.format(self.degradation))
        if self.scale.is_integer():
            path_lr = path.join(path_lr, 'X{}'.format(int(self.scale)))
        else:
            path_lr = path.join(path_lr, 'X{:.2f}'.format(self.scale))

        scan_dict = {'hr': path_hr, 'lr': path_lr}
        return scan_dict

    def scan_rule(self, scan_dir: str, k: str) -> list:
        scan_list = glob.glob(path.join(scan_dir, '*.png'))
        return scan_list

    def scan(self) -> dict:
        scan_dict = self.scan_dirs().items()
        scan_dict = {k: sorted(self.scan_rule(v, k)) for k, v in scan_dict}
        return scan_dict

    def load_file(self, x: str) -> np.array:
        #return imageio.imread(x)
        return self.cache(x)
