import os
from os import path
import glob
import random
import types
import typing
import pickle

from data import common

import imageio
from srwarp import wtypes

import torch
import tqdm

class WarpData(torch.utils.data.Dataset):

    def __init__(
            self,
            data_path: str=None,
            bin_path: str=None,
            m_path: str=None,
            is_binary: bool=True,
            preprocessing: common.Preprocessing=None,
            train: bool=True) -> None:

        self.imgs = sorted(glob.glob(path.join(data_path, '*.png')))
        self.ms = torch.load(m_path)
        if 'patch_size' in self.ms:
            self.patch_size = self.ms['patch_size']
        else:
            self.patch_size = None

        if train:
            self.ms = self.ms['train']
        else:
            self.ms = self.ms['eval']

        if is_binary:
            if bin_path is None:
                bin_path = data_path

            b = path.join(bin_path, 'bin')
            if bin_path in data_path:
                bin_path = data_path.replace(bin_path, b)
            else:
                bin_path = b

            bin_ext = 'pt'
            os.makedirs(bin_path, exist_ok=True)
            for img in tqdm.tqdm(self.imgs, ncols=80):
                name = path.basename(img)
                bin_name = name.replace('png', bin_ext)
                bin_name = path.join(bin_path, bin_name)
                if not path.isfile(bin_name):
                    x = imageio.imread(img)
                    with open(bin_name, 'wb') as f:
                        pickle.dump(x, f)

            self.imgs = sorted(glob.glob(path.join(bin_path, '*.' + bin_ext)))

        self.is_binary = True
        self.preprocessing = preprocessing
        self.train = train
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace, train: bool=True) -> dict:
        kwargs = {
            'bin_path': cfg.bin_path,
            'm_path': cfg.m_path,
            'preprocessing': common.make_pre(cfg),
            'train': train,
        }
        if train:
            kwargs['data_path'] = cfg.data_path_train
        else:
            kwargs['data_path'] = cfg.data_path_test

        return kwargs

    def __getitem__(self, idx: int) -> wtypes._TT:
        name = self.imgs[idx]
        if self.is_binary:
            with open(name, 'rb') as f:
                img = pickle.load(f)
        else:
            img = imageio.imread(name)

        h, w, _ = img.shape
        if self.patch_size is not None:
            if self.train:
                px = random.randrange(0, w - self.patch_size + 1)
                py = random.randrange(0, h - self.patch_size + 1)
            else:
                px = (w - self.patch_size) // 2
                py = (h - self.patch_size) // 2

            img = img[py:(py + self.patch_size), px:(px + self.patch_size)]

        if self.train:
            img_dict = self.preprocessing.augment(img=img)
        else:
            img_dict = {'img': img}

        img_dict = self.preprocessing.set_color(**img_dict)
        img_dict = self.preprocessing.np2Tensor(**img_dict)

        if self.train:
            m_idx = random.randrange(len(self.ms))
        else:
            m_idx = idx % 100

        img_dict['m'] = self.ms[m_idx].double()
        img_dict['m_idx'] = m_idx
        img_dict['name'] = path.splitext(path.basename(name))[0]
        return img_dict

    def __len__(self) -> int:
        return len(self.imgs)
