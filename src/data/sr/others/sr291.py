from os import path
import glob

from data.sr import base


class SR291(base.SRBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_path(self) -> str:
        return path.join(self.dpath, 'SR291')
