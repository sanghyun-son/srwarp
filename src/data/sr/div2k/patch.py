import os
from os import path
import glob
import random
import imageio

from data import common
from torch import utils


class DIV2KPatch(utils.data.Dataset):

    def __init__(
            self, dpath=None, scale=4, preprocessing=None,
            train_range='0-100', unpaired=False, no_repeat=True, train=True):

        self.unpaired = unpaired
        self.no_repeat = no_repeat
        self.pre = preprocessing
        self.train = train

        dpath = self.get_path(dpath, train)
        self.scan_data(dpath, scale, train_range)

    def get_path(self, dpath, train):
        if train:
            dpath = path.join(dpath, 'DIV2K', 'patch_importance')
        else:
            dpath = path.join(dpath, 'DIV2K', 'patch_importance_valid')

        return dpath

    def scan_data(self, dpath, scale, train_range):
        metadata = path.join(dpath, 'metadata.txt')
        with open(metadata, 'r') as f:
            lines = f.read().splitlines()
            lines = [line.split(' ') for line in lines]
            # filename / original / y / x / std
            parse = lambda x, y, z, w, v: {'filename': x, 'std': float(v)}
            lines = [parse(*line) for line in lines]
            lines.sort(key=lambda x: x['std'], reverse=True)

        self.dpath = dpath
        self.dir_hr = path.join(dpath, 'HR')
        if self.unpaired and self.train:
            self.dir_lr = self.dir_hr
        else:
            self.dir_lr = path.join(dpath, 'LR', 'X{}'.format(scale))

        # Crop a subset from the full dataset
        low, high = [
            int(len(lines) * float(r)) // 100 for r in train_range.split('-')
        ]
        self.lines = lines[low:high]
        if self.no_repeat:
            self.len_full = len(self.lines)
        else:
            self.len_full = len(lines)

    @staticmethod
    def get_kwargs(cfg, train=True):
        kwargs = {
            'dpath': cfg.dpath,
            'scale': cfg.scale,
            'preprocessing': common.make_pre(cfg),
            'train_range': cfg.train_range,
            'unpaired': cfg.unpaired,
            'train': train,
        }
        return kwargs

    def __len__(self):
        if self.train:
            return self.len_full
        else:
            return len(self.lines)

    def __getitem__(self, idx):
        idx = idx % len(self.lines)
        filename_hr = self.lines[idx]['filename']
        if self.unpaired and self.train:
            while True:
                idx_lr = random.choice(range(len(self)))
                if idx_lr != idx:
                    filename_lr = self.lines[idx_lr]['filename']
                    break
        else:
            filename_lr = filename_hr

        img_dict = {
            'lr': imageio.imread(path.join(self.dir_lr, filename_lr)),
            'hr': imageio.imread(path.join(self.dir_hr, filename_hr)),
        }
        if self.train:
            img_dict = self.pre.augment(**img_dict)

        img_dict = self.pre.np2Tensor(**img_dict)
        return img_dict

