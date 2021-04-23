from os import path

from data import common
from data.sr import dataclass

_parent_class = dataclass.SRData

class DIV2K(_parent_class):
    '''
    DIV2K mean:
        R: 0.4488
        G: 0.4371
        B: 0.4040
    '''

    def __init__(self, *args, dbegin=1, dend=100, tval=False, **kwargs):
        self.dbegin = dbegin
        self.dend = dend
        self.tval = tval
        super(DIV2K, self).__init__(*args, **kwargs)

    @staticmethod
    def get_kwargs(cfg, train=True):
        kwargs = _parent_class.get_kwargs(cfg, train=train)
        if train:
            dbegin = 1
            dend = 400
        else:
            kwargs['tval'] = ('t' in cfg.val_range)
            val_range = cfg.val_range.replace('t', '')
            dbegin, dend = [int(x) for x in val_range.split('-')]

        kwargs['dbegin'] = dbegin
        kwargs['dend'] = dend
        return kwargs

    def scan(self, target_path):
        filelist = super(DIV2K, self).scan(target_path)
        if self.dbegin and self.dend:
            filelist = filelist[self.dbegin - 1:self.dend]

        return filelist

    def apath(self):
        return path.join(self.dpath, 'DIV2K')

    def get_path(self, degradation, scale):
        if scale.is_integer():
            scale = int(scale)

        if not (self.train or self.tval):
            split = 'valid'
        else:
            split = 'train'

        path_hr = path.join(self.apath(), 'DIV2K_{}_HR'.format(split))

        if scale == 1:
            path_lr = path_hr
        else:
            if 'jit' in degradation:
                path_lr = degradation
            else:
                path_lr = 'DIV2K_{}_LR_{}'.format(split, degradation)
                path_lr = path.join(self.apath(), path_lr, 'X{}'.format(scale))

        return {'lr': path_lr, 'hr': path_hr}

