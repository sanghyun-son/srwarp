from os import path

from data import common
from data.sr import dataclass

_parent_class = dataclass.SRData


class Custom(_parent_class):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apath(self):
        return path.join(self.dpath, 'Custom')

    def get_path(self, degradation, scale):
        path_hr = path.join(self.apath(), degradation, 'HR')
        path_lr = path.join(self.apath(), degradation, 'LR')
        return {'lr': path_lr, 'hr': path_hr}

