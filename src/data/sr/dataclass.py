import os
from os import path
import sys
import glob
import warnings
import pickle
import itertools

from data import common
from data import sampler

import numpy as np
import imageio
from torch import utils
import tqdm


class SRData(utils.data.Dataset):

    def __init__(
            self, degradation='bicubic', scale=4, preprocessing=None,
            dpath=None, is_binary=True, force_ram=False,
            min_size=256, train=True, **kwargs):

        if not isinstance(degradation, (list, tuple)):
            degradation = (degradation,)

        if not isinstance(scale, (list, tuple)):
            scale = (scale,)

        self.degradation = degradation
        self.scale = scale
        self.pre = preprocessing
        self.dpath = dpath
        self.min_size = min_size
        self.train = train
        self.force_ram = force_ram
        '''
        self.data is a dictionary of file list

        Example::
            >>> print(self.data)
            {'lr': [lr_1.png, lr_2.png, ...], 'hr': [hr_1.png, hr_2.png, ...]}
        '''
        self.data = self.collect_files()
        # We will not use EXIF data
        warnings.filterwarnings('ignore', message='Corrupt EXIF data')
        warnings.filterwarnings('ignore', message='Possibly corrupt EXIF data')
        if is_binary:
            print('Making binaries...')
            if force_ram:
                print('Loading all binaries to memory...')

            self.data = {k: self.to_bin(v) for k, v in self.data.items()}

    @staticmethod
    def get_kwargs(cfg, train=True):
        dpath = cfg.dpath
        degradation = cfg.degradation
        if not train:
            if cfg.dpath_test is not None:
                dpath = cfg.dpath_test

            if cfg.degradation_test is not None:
                degradation = cfg.degradation_test

        kwargs = {
            'degradation': degradation,
            'scale': cfg.scale,
            'preprocessing': common.make_pre(cfg),
            'dpath': dpath,
            'is_binary': not cfg.raw,
            'force_ram': cfg.force_ram,
            'min_size': cfg.scale * cfg.patch,
            'train': train,
        }
        return kwargs

    # Below functions are used to prepare images
    def scan(self, target_path):
        '''
        This function defines a scan rule.

        Args:
            target_path (path-like object):

        Return:
            list:
        '''
        exts = ['.png', '.jpg', '.jpeg', '.bmp']
        exts.extend([ext.upper() for ext in exts])
        files = []
        for d, s, _ in os.walk(target_path):
            if not s:
                for ext in exts:
                    files.extend(glob.glob(path.join(d, '*' + ext)))

        files.sort()
        return files

    def collect_files(self):
        '''
        Scan all available samples.

        Return:
        '''
        paths = {}
        def add_path(new_path):
            for k, v in new_path.items():
                if isinstance(v, (list, tuple)):
                    v = list(v)
                else:
                    v = [v]

                if k in paths:
                    paths[k].extend(v)
                else:
                    paths[k] = v

        for d, s in itertools.product(self.degradation, self.scale):
            add_path(self.get_path(d, s))

        scan_list = lambda x: sum([self.scan(p) for p in x], [])
        file_paths = {k: scan_list(v) for k, v in paths.items()}
        return file_paths

    def apath(self):
        return self.dpath

    def get_path(self, degradation, scale):
        '''
        Get a path to the dataset of given scale and degradation operators.

        Args:

        Return:
        '''
        path_hr = path.join(self.apath(), 'HR')
        if scale == 1:
            path_lr = path_hr
        else:
            path_lr = path.join(self.apath(), 'LR_{}'.format(degradation))
            path_lr = path.join(path_lr, 'X{}'.format(scale))

        path_dict = {'lr': path_lr, 'hr': path_hr}
        return path_dict

    def to_bin(self, files):
        compressed = ['.jpg', '.jpeg']
        compressed.extend([comp.upper() for comp in compressed])
        bin_files = []
        tq = tqdm.tqdm(files, ncols=80)
        for f in tq:
            tq.set_description('{: <15}'.format(path.basename(f)))
            '''
            For files with high compression ratio,
            it is inconvinient to store them in binary format.
            '''
            if path.splitext(f)[-1] in compressed:
                img = imageio.imread(f)
                h = img.shape[0]
                w = img.shape[1]
                if h >= self.min_size and w >= self.min_size:
                    if self.force_ram:
                        bin_files.append({'name': f, 'img': img})
                    else:
                        bin_files.append(f)
            else:
                bin_file = self.to_bin_name(f)
                ret = self.make_bin(f, bin_file)
                if self.force_ram:
                    bin_files.append({'name': bin_file, 'img': ret})
                else:
                    bin_files.append(bin_file)


        return bin_files

    def to_bin_name(self, name):
        apath = self.apath()
        bin_name = name.replace(apath, path.join(apath, 'bin'))
        ext = path.splitext(name)[-1]
        if ext:
            bin_name = bin_name.replace(ext, '.pt')

        return bin_name

    def make_bin(self, name, bin_name):
        if path.isfile(bin_name):
            if self.force_ram:
                with open(bin_name, 'rb') as f:
                    return pickle.load(f)
            else:
                return None

        os.makedirs(path.dirname(bin_name), exist_ok=True)
        with open(bin_name, 'wb') as f:
            img = imageio.imread(name)
            pickle.dump(img, f)
            if self.force_ram:
                return img

            return None

    def __getitem__(self, idx):
        # SR data preparation pipeline
        idx_dict = self.get_idx(idx)
        file_dict = self.get_filename(**idx_dict)
        img_dict = self.get_file(**file_dict)
        img_dict = self.get_patch(**img_dict)
        img_dict = self.pre.set_color(**img_dict)
        if self.train:
            img_dict = self.pre.compress(**img_dict)

        img_dict = self.pre.np2Tensor(**img_dict)
        img_dict = self.pre.add_noise(**img_dict)
        if not self.train:
            # we also provide filename during evaluation
            if self.force_ram:
                name = self.data['hr'][idx_dict['hr']]['name']
            else:
                name = file_dict['hr']

            name, _ = path.splitext(path.basename(name))
            img_dict['name'] = name

        return img_dict

    def __len__(self):
        '''
        Return:
            int: the number of train images (not patches!)

        Note:
            Since we have a different definition for 1 epoch,
            please refer to data.sampler for more detail.
        '''
        return len(self.data['hr'])

    def get_idx(self, idx):
        '''
        Return:
            dict: indices for train pairs

        Note:
            For unpaired dataset, we can override this function to
            return a tuple of 'unpaired' train samples instead.
        '''
        idx_dict = {k: idx for k in self.data.keys()}            
        return idx_dict

    def get_filename(self, **kwargs):
        if self.force_ram:
            return kwargs
        else:
            file_dict = {k: self.data[k][v] for k, v in kwargs.items()}
            return file_dict

    def get_file(self, **kwargs):
        def read(x):
            if '.pt' in x:
                with open(x, 'rb') as f:
                    return pickle.load(f)
            else:
                return imageio.imread(x)

        if self.force_ram:
            img_dict = {k: self.data[k][v]['img'] for k, v in kwargs.items()}
        else:
            img_dict = {k: read(v) for k, v in kwargs.items()}

        return img_dict

    def get_patch(self, **kwargs):
        img_dict = self.pre.modcrop(**kwargs)
        if self.train:
            img_dict = self.pre.get_patch(**img_dict)
            img_dict = self.pre.augment(**img_dict)
            img_dict = self.pre.add_noise(**img_dict)

        return img_dict

