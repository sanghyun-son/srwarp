import os
from os import path
import sys
import shutil
import datetime

import tqdm

from torch.utils import tensorboard

_print = tqdm.tqdm.write


class Logger():
    '''

    '''

    def __init__(self, cfg):
        self.path = path.join('..', 'experiment', cfg.save, cfg.ablation)
        if cfg.reset:
            if path.isdir(self.path):
                response = input(
                    'Do you want to remove the existing directory? [Y/N]: '
                )
                is_reset = (response.lower() == 'y')
            else:
                is_reset = True

            if is_reset:
                shutil.rmtree(self.path, ignore_errors=True)

        os.makedirs(self.path, exist_ok=True)
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        with open(self.get_path('config.txt'), 'a') as f:
            # save a command line and all arguments
            f.write(now + '\n')
            f.write('python ' + ' '.join(sys.argv) + '\n\n')
            for k, v in vars(cfg).items():
                f.write('{}: {}\n'.format(k, v))

            f.write('-' * 80 + '\n\n')

        with open('global.txt', 'a') as f:
            f.write('{}:\t{}\n'.format(now, self.path))

        self.writer = tensorboard.SummaryWriter(log_dir=self.path)

    def __enter__(self):
        self.open_log()
        return self

    def __exit__(self, *args, **kwargs):
        self.log_file.close()

    def __call__(self, obj, display=True, refresh=False, clip=0, filename=None):
        if display:
            if isinstance(obj, (list, tuple)):
                for s in obj:
                    _print(s)

                obj = '\n'.join(obj)
            else:
                obj = str(obj)
                if clip > 0:
                    clip_obj = obj.splitlines()
                    clip_obj = clip_obj[:clip] + ['...']
                    clip_obj = '\n'.join(clip_obj)
                    _print(clip_obj)
                else:
                    _print(obj)

        if not filename:
            if self.log_file is None:
                return

            try:
                self.log_file.write(obj + '\n')
            except Exception as e:
                _print('Cannot write log!')
                _print(e)
                self.log_file = None
                return

            if refresh:
                try:
                    self.log_file.flush()
                except Exception as e:
                    _print('An error occured on log.txt!')
                    _print(e)
                    self.log_file = None
                    return
        else:
            with open(self.get_path(filename), 'w') as f:
                f.write(obj + '\n')

    def get_path(self, *subdirs):
        return path.join(self.path, *subdirs)

    def open_log(self, save_as=None):
        try:
            if save_as is None:
                save_as = 'log.txt'

            self.log_file = open(self.get_path(save_as), 'a')
        except Exception as e:
            _print('Cannot open log.txt!')
            _print(e)
            self.log_file = None
