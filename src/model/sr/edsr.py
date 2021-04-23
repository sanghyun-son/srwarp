from config import get_config
from model import common

from torch import nn


class EDSR(nn.Module):
    '''
    EDSR model

    Note:
        From Lim et al.,
        "Enhanced Deep Residual Networks for Single Image Super-Resolution"
        See https://arxiv.org/pdf/1707.02921.pdf for more detail.
    '''

    def __init__(
            self,
            scale=4, depth=16, n_colors=3, n_feats=64,
            res_scale=1, res_prob=1, conv=common.default_conv):

        super(EDSR, self).__init__()
        self.n_colors = n_colors
        self.conv = conv(n_colors, n_feats, 3)
        resblock = lambda: common.ResBlock(
            n_feats, 3, conv=conv, res_scale=res_scale, res_prob=res_prob,
        )
        m = [resblock() for _ in range(depth)]
        m.append(conv(n_feats, n_feats, 3))
        self.resblocks = nn.Sequential(*m)
        self.recon = nn.Sequential(
            common.Upsampler(scale, n_feats, conv=conv),
            conv(n_feats, n_colors, 3),
        )
        self.url = None

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        parse_list = [
            'scale', 'depth', 'n_colors', 'n_feats', 'res_scale', 'res_prob'
        ]
        kwargs = get_config.parse_namespace(cfg, *parse_list)
        kwargs['conv'] = conv
        return kwargs

    def forward(self, x):
        if self.n_colors == 1:
            B, C, H, W = x.size()
            x = x.view(B * C, 1, H, W)
        x = self.conv(x)
        x = x + self.resblocks(x)
        x = self.recon(x)
        if self.n_colors == 1:
            _, _, H, W = x.size()
            x = x.view(B, C, H, W)
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for k in own_state.keys():
            if k not in state_dict and 'recon' not in k:
                raise RuntimeError(k + ' does not exist!')
            else:
                if k in state_dict:
                    own_state[k] = state_dict[k]

        super().load_state_dict(own_state, strict=strict)

