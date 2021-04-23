import importlib
import os
from os import path
import sys
import time
import math
import collections
import multiprocessing
import imageio
import pickle
import itertools

import torch
import torchvision.utils as vutils


def str2num(string):
    '''
    Extract a number from the given string

    Arg:
    '''
    return int(''.join(s for s in string if s.isdigit()))


def format_vp(n):
    if n == 0:
        return '0'
    elif abs(n) < 1e-4:
        return '{:.1g}'.format(n)
    else:
        log_len = int(min(4, max(0, math.log10(abs(n)))))
        return '{:.{}f}'.format(n, 4 - log_len)


def save_with_exception(obj, name):
    try:
        dirname = path.dirname(name)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(obj, name)
    except IOError as io_error:
        print('Fail to save {}!'.format(name))
        print(io_error)
    except Exception as e:
        print(e)

def to(obj, gpus='auto', precision='single'):
    '''
    A simple and convinient wrapper for torch.to()
    '''
    if gpus == 'auto':
        if torch.cuda.is_available():
            gpus = 1
        else:
            gpus = 0

    if gpus > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if precision == 'half':
        dtype = torch.half
    else:
        dtype = None

    return obj.to(device=device, dtype=dtype)


class SRMisc(object):
    '''
    Miscellaneous things for super-resolution

    '''

    def __init__(self, cfg=None):
        self.queue = None
        if cfg is not None:
            self.print_every = cfg.print_every
        else:
            self.print_every = 100

    def is_print(self, batch):
        return (batch + 1) % self.print_every == 0

    def quantize(self, *xs):
        '''
        Quantize -1 ~ 1 to 0 ~ 255.

        Args:

        Return:

        '''
        rgb = 255
        _quantize = lambda x: x.add(1).mul(rgb / 2).round().clamp(0, rgb)
        if len(xs) == 1:
            return _quantize(xs[0])
        else:
            return [_quantize(x) for x in xs]

    def begin_background(self):
        self.queue = multiprocessing.Queue()

        def target(queue):
            while True:
                if queue.empty():
                    continue

                name, x = queue.get()
                if name is None:
                    return

                try:
                    if '.pt' in name:
                        with open(name, 'wb') as f:
                            pickle.dump(x, f)
                    elif '.jp' in name or '.JP' in name:
                        imageio.imwrite(name, x, quality=100)
                    else:
                        imageio.imwrite(name, x)
                except IOError as io_error:
                    print('Cannot save image file!')
                    print(io_error)
                except Exception as e:
                    print(e)
                    sys.exit(1)

        def worker():
            return multiprocessing.Process(target=target, args=(self.queue,))

        self.process = [worker() for _ in range(multiprocessing.cpu_count())]
        for p in self.process:
            p.start()

    def end_background(self):
        if self.queue is None:
            return

        for _ in self.process:
            self.queue.put((None, None))

    def join_background(self):
        if self.queue is None:
            return

        while not self.queue.empty():
            time.sleep(0.5)

        for p in self.process:
            p.join()

        self.queue = None

    @staticmethod
    def tensor2np(x):
        x = x.permute(1, 2, 0)
        x = x.byte().cpu().numpy()
        return x

    def save(self, x, save_as, name, exts='.png', single=False):
        '''
        Save output result as images

        Args:
            x (dict or Tensor): an image, or images with their keys
            save_as (str): name of the subdirectory
            name (str): name of the file
        '''
        if not isinstance(exts, (list, tuple)):
            exts = (exts,)

        y = {}
        for ext in exts:
            if isinstance(x, dict):
                for k ,v in x.items():
                    y['{}_{}{}'.format(name, k, ext)] = v
            else:
                y['{}{}'.format(name, ext)] = x

        os.makedirs(save_as, exist_ok=True)
        if self.queue is None:
            try:
                self.begin_background()
            except Exception as e:
                print('Cannot start threads!')
                print(e)
                return

        for k, v in y.items():
            name = path.join(save_as, k)
            v = self.quantize(v)
            if v.size(0) > 1:
                vutils.save_image(v / 255, name, nrow=16, padding=0)
            else:
                v = SRMisc.tensor2np(v[0])
                self.queue.put((name, v))

        if single:
            self.end_background()
            self.join_background()
