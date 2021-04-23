import math
import typing

from config import get_config
from model import common

import torch
from torch import nn


class MSEDSR(nn.Module):
    '''
    Multi-scale EDSR model
    '''

    def __init__(
            self,
            max_scale: int=4,
            depth: int=16,
            n_colors: int=3,
            n_feats: int=64,
            model_flag: typing.Optional[str]=None,
            conv=common.default_conv):

        super().__init__()
        self.n_colors = n_colors
        self.model_flag = model_flag

        self.conv_in = conv(n_colors, n_feats, 3)
        m = [common.ResBlock(n_feats, 3, conv=conv) for _ in range(depth)]
        m.append(conv(n_feats, n_feats, 3))
        self.resblocks = nn.Sequential(*m)
        self.up_img = common.Upsampler(2, n_colors, bias=False, conv=conv)
        self.up_feat = common.Upsampler(2, n_feats, conv=conv)
        self.conv_out = conv(n_feats, n_colors, 3)

        self.n_scales = int(math.log2(max_scale))

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        parse_list = ['depth', 'n_colors', 'n_feats', 'model_flag']
        kwargs = get_config.parse_namespace(cfg, *parse_list)
        kwargs['max_scale'] = cfg.scale
        kwargs['conv'] = conv
        return kwargs

    def forward(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        output_list = []
        f = self.conv_in(x)
        for _ in range(self.n_scales):
            f = f + self.resblocks(f)
            f = self.up_feat(f)
            x = self.conv_out(f) + self.up_img(x)
            output_list.append(x)

        if self.model_flag == 'no_multi':
            output_list = output_list[-1]

        return output_list
