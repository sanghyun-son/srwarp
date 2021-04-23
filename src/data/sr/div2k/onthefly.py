from os import path
import types
import typing

from data.sr.div2k import base

_parent_class = base.DIV2K


class DIV2KOnTheFly(_parent_class):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return

    @staticmethod
    def get_kwargs(
            cfg: types.SimpleNamespace,
            train: bool=True) -> typing.Mapping[str, object]:

        kwargs = _parent_class.get_kwargs(cfg, train=train)
        kwargs['preprocessing'].patch += 2 * cfg.max_scale
        return kwargs

    def get_path(
            self,
            degradation: str,
            scale: int) -> typing.Mapping[str, str]:

        if not (self.train or self.tval):
            split = 'valid'
        else:
            split = 'train'

        path_hr = path.join(self.apath(), 'DIV2K_{}_HR'.format(split))
        path_dict = {'hr': path_hr}
        return path_dict
