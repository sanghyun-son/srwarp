import collections

from config import get_config
from model import common

from torch import nn
from torch.nn import functional as F


class DDBPN(nn.Module):

    def __init__(
            self, scale=4, depth=3, n_colors=3, n_feats=64,
            conv=common.default_conv):

        super().__init__()
        if scale == 2:
            kernel_size = 6
            kwargs = {'stride': 2, 'padding': 2}
        elif scale == 4:
            kernel_size = 8
            kwargs = {'stride': 4, 'padding': 2}
        elif scale == 8:
            kernel_size = 12
            kwargs = {'stride': 8, 'padding': 2}

        self.scale = scale
        self.depth = depth
        self.n_feats = n_feats

        norm = None
        act = 'prelu'
        kwargs['norm'] = norm
        kwargs['act'] = act

        self.convs = nn.Sequential(
            common.BasicBlock(n_colors, 4 * n_feats, 3, norm=norm, act=act),
            common.BasicBlock(4 * n_feats, n_feats, 1, norm=norm, act=act),
        )
        self.bp = nn.ModuleDict(modules=collections.OrderedDict(
            up_1=UpBlock(n_feats, kernel_size, **kwargs),
            down_1=DownBlock(n_feats, kernel_size, **kwargs),
            up_2=UpBlock(n_feats, kernel_size, **kwargs),
            down_2=DownBlock(n_feats, kernel_size, stage=2, **kwargs),
            up_3=UpBlock(n_feats, kernel_size, stage=2, **kwargs),
            down_3=DownBlock(n_feats, kernel_size, stage=3, **kwargs),
            up_4=UpBlock(n_feats, kernel_size, stage=3, **kwargs),
            down_4=DownBlock(n_feats, kernel_size, stage=4, **kwargs),
            up_5=UpBlock(n_feats, kernel_size, stage=4, **kwargs),
            down_5=DownBlock(n_feats, kernel_size, stage=5, **kwargs),
            up_6=UpBlock(n_feats, kernel_size, stage=5, **kwargs),
            down_6=DownBlock(n_feats, kernel_size, stage=6, **kwargs),
            up_7=UpBlock(n_feats, kernel_size, stage=6, **kwargs),
        ))
        self.recon = conv(depth * n_feats, n_colors, 3)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        parse_list = ['scale', 'n_colors', 'n_feats']
        kwargs = get_config.parse_namespace(cfg, *parse_list)
        kwargs['conv'] = conv
        return kwargs

    def forward(self, x):
        # To make buffers
        b, _, h, w = x.size()

        prev_iter = self.convs(x)
        buffer_res = x.new_zeros(
            b, self.depth * self.n_feats, self.scale * h, self.scale * w
        )
        for cr in range(self.depth):
            buffer_h = x.new_zeros(
                b, 6 * self.n_feats, self.scale * h, self.scale * w
            )
            buffer_l = x.new_zeros(b, 6 * self.n_feats, h, w)
            ch = 0
            cl = 0

            for k, v in self.bp.items():
                if k == 'up_1':
                    buffer_h[:, ch:ch + self.n_feats] = v(prev_iter)
                    ch += self.n_feats
                elif k == 'down_1':
                    h_1 = buffer_h[:, ch - self.n_feats:ch]
                    buffer_l[:, cl:cl + self.n_feats] = v(h_1)
                    cl += self.n_feats
                elif k == 'up_2':
                    l_1 = buffer_l[:, cl - self.n_feats:cl]
                    buffer_h[:, ch:ch + self.n_feats] = v(l_1)
                    ch += self.n_feats
                elif k == 'up_7':
                    acc = cr * self.n_feats
                    buffer_res[:, acc:acc + self.n_feats] = v(buffer_l)
                elif 'down' in k:
                    h_x = buffer_h[:, :ch]
                    buffer_l[:, cl:cl + self.n_feats] = v(h_x)
                    if k == 'down_6':
                        prev_iter = buffer_l[:, cl:cl + self.n_feats]
                    cl += self.n_feats
                else:
                    l_x = buffer_l[:, :cl]
                    buffer_h[:, ch:ch + self.n_feats] = v(l_x)
                    ch += self.n_feats

        x = F.interpolate(
            x,
            scale_factor=self.scale,
            align_corners=False,
            mode='bicubic',
        )
        x = x + self.recon(buffer_res)
        return x


class UpBlock(nn.Module):
    
    def __init__(
            self, n_feats, kernel_size,
            stride=4, padding=2, stage=None, norm=None, act='prelu'):

        super().__init__()
        args = [n_feats, n_feats, kernel_size]
        kwargs = {
            'stride': stride, 'padding': padding, 'norm': norm, 'act': act,
        }
        if stage is None:
            self.conv = None
        else:
            # For dense connection
            self.conv = common.BasicBlock(
                stage * n_feats, n_feats, 1, norm=norm, act=act,
            )

        self.up_1 = common.BasicTBlock(*args, **kwargs)
        self.up_2 = common.BasicBlock(*args, **kwargs)
        self.up_3 = common.BasicTBlock(*args, **kwargs)

    def forward(self, x):
        # Bottleneck
        if self.conv:
            x = self.conv(x)

        h_0 = self.up_1(x)
        l_0 = self.up_2(h_0)
        h_1 = self.up_3(l_0 - x)
        ret = h_0 + h_1
        return ret


class DownBlock(nn.Module):
    
    def __init__(
            self, n_feats, kernel_size,
            stride=4, padding=2, stage=None, norm=None, act='prelu'):

        super().__init__()
        args = [n_feats, n_feats, kernel_size]
        kwargs = {
            'stride': stride, 'padding': padding, 'norm': norm, 'act': act,
        }
        if stage is None:
            self.conv = None
        else:
            # For dense connection
            self.conv = common.BasicBlock(
                stage * n_feats, n_feats, 1, norm=norm, act=act,
            )

        self.down_1 = common.BasicBlock(*args, **kwargs)
        self.down_2 = common.BasicTBlock(*args, **kwargs)
        self.down_3 = common.BasicBlock(*args, **kwargs)

    def forward(self, x):
        # Bottleneck
        if self.conv:
            x = self.conv(x)

        l_0 = self.down_1(x)
        h_0 = self.down_2(l_0)
        l_1 = self.down_3(h_0 - x)
        ret = l_0 + l_1
        return ret

# For loading the module
REPRESENTATIVE = DDBPN

