from os import path
from data.sr import base


class B100(base.SRBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return

    def get_path(self) -> str:
        return path.join(self.dpath, 'benchmark', 'b100')

