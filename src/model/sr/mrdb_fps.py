from model import common
from config import get_config

import torch
from torch import nn
from torch.nn import functional


class RDBlock(nn.Module):

    def __init__(
            self,
            n_feats: int,
            gf: int,
            drep: int=5,
            res_scale: float=0.2,
            conv=common.default_conv) -> None:

        super().__init__()
        self.n_feats = n_feats
        self.n_max = n_feats + (drep - 1) * gf
        self.gf = gf
        self.res_scale = res_scale
        kwargs = {'padding_mode': 'reflect'}
        m = [conv(n_feats + i * gf, gf, 3, **kwargs) for i in range(drep - 1)]
        m.append(conv(self.n_max, n_feats, 3, **kwargs))
        self.convs = nn.ModuleList(m)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        We will not use torch.cat since it consumes GPU memory lot.
        '''
        b, _, h, w = x.size()
        c = self.n_feats
        for i, conv in enumerate(self.convs):
            x_inc = conv(x)
            if i == len(self.convs) - 1:
                return x[:, :self.n_feats] + self.res_scale * x_inc
            else:
                x_inc = functional.leaky_relu(
                    x_inc, negative_slope=0.2, inplace=True
                )
                x = torch.cat((x, x_inc), dim=1)


class RRDBlock(nn.Sequential):

    def __init__(
            self,
            n_feats: int,
            gf: int,
            rep: int=3,
            res_scale: float=0.2,
            conv=common.default_conv) -> None:

        self.res_scale = res_scale
        args = [n_feats, gf]
        kwargs = {'res_scale': res_scale, 'conv': conv}
        m = [RDBlock(*args, **kwargs) for _ in range(rep)]
        super().__init__(*m)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.res_scale * super().forward(x)
        return x


class MRDBFPS(nn.Module):
    '''
    RRDB model

    Note:
        From
        "ESRGAN"
        See for more detail.
    '''

    def __init__(
            self,
            depth: int=23,
            n_colors: int=3,
            n_feats: int=64,
            gf: int=32,
            res_scale: float=0.2,
            multi_scale: bool=True,
            conv: nn.Module=common.default_conv) -> None:

        super().__init__()
        kwargs = {'padding_mode': 'reflect'}
        self.conv = conv(n_colors, n_feats, 3, **kwargs)
        block = lambda: RRDBlock(n_feats, gf, res_scale=res_scale)
        m = [block() for _ in range(depth)]
        m.append(conv(n_feats, n_feats, 3, **kwargs))
        self.rrdblocks = nn.Sequential(*m)
        self.recon_x1 = conv(n_feats, n_feats, 3, **kwargs)
        if multi_scale:
            self.recon_x2 = common.Upsampler(2, n_feats, **kwargs)
            self.recon_x4 = common.Upsampler(4, n_feats, **kwargs)

        self.multi_scale = multi_scale
        return

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        parse_list = ['n_colors', 'n_feats']
        kwargs = get_config.parse_namespace(cfg, *parse_list)
        kwargs['gf'] = cfg.n_feats // 2
        kwargs['depth'] = 23
        kwargs['res_scale'] = 0.2
        kwargs['conv'] = conv
        return kwargs

    def forward(self, x):
        x = self.conv(x)
        x = x + self.rrdblocks(x)
        x_1 = self.recon_x1(x)

        if self.multi_scale:
            x_2 = self.recon_x2(x)
            x_4 = self.recon_x4(x)
            return x_1, x_2, x_4
        else:
            return x_1


# For loading the module
REPRESENTATIVE = MRDBFPS

