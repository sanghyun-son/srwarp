from os import path
from data.sr import base


class Urban100(base.SRBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_path(self) -> str:
        return path.join(self.dpath, 'benchmark', 'urban100')

