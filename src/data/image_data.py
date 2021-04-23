import os
from os import path
import glob
import random
import types
import typing
import pickle

from data import common

import imageio

import torch
import tqdm

class ImageData(torch.utils.data.Dataset):

    def __init__(
            self,
            data_path: str=None,
            bin_path: str=None,
            is_binary: bool=True,
            preprocessing: typing.Optional[common.Preprocessing]=None,
            train: bool=True) -> None:

        self.imgs = glob.glob(path.join(data_path, '*.png'))

        if is_binary:
            base = path.basename(data_path)
            if bin_path is None:
                bin_path = data_path.replace(base, path.join('bin', base))
            else:
                bin_path = path.join(bin_path, 'bin', base)

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

            self.imgs = glob.glob(path.join(bin_path, '*.' + bin_ext))

        self.is_binary = True
        self.preprocessing = preprocessing
        self.train = train
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace, train=True) -> dict:
        kwargs = {
            'bin_path': cfg.bin_path,
            'preprocessing': common.make_pre(cfg),
            'train': train,
        }
        if train:
            kwargs['data_path'] = cfg.data_path_train
        else:
            kwargs['data_path'] = cfg.data_path_test

        return kwargs

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        name = self.imgs[idx]
        if self.is_binary:
            with open(name, 'rb') as f:
                img = pickle.load(f)
        else:
            img = imageio.imread(name)

        if self.train:
            img_dict = self.preprocessing.get_patch(img=img)
            img_dict = self.preprocessing.augment(**img_dict)
        else:
            img_dict = {'img': img}

        img_dict = self.preprocessing.set_color(**img_dict)
        img_dict = self.preprocessing.np2Tensor(**img_dict)
        img_dict['name'] = path.splitext(path.basename(name))[0]
        return img_dict

    def __len__(self) -> int:
        return len(self.imgs)
