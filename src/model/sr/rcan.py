## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
from torch import nn

def model_class(cfg=None, conv=common.default_conv, make=False):
    if make:
        return RCAN(**RCAN.get_kwargs(cfg=cfg, conv=conv))
    else:
        return RCAN


## Channel Attention (CA) Layer
class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.conv_du(x)
        return x


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction):

        super(RCAB, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=True))
            if i == 0:
                m.append(nn.ReLU(True))

        m.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, depth):
        super(ResidualGroup, self).__init__()
        rcab = lambda: RCAB(conv, n_feat, kernel_size, reduction)
        m = [rcab() for _ in range(depth)]
        m.append(conv(n_feat, n_feat, kernel_size))
        self.rcabs = nn.Sequential(*m)

    def forward(self, x):
        x = x + self.rcabs(x)
        return x


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):

    def __init__(
            self,
            scale=4, depth=20, n_colors=3, n_feats=64,
            n_resgroups=10, reduction=16, conv=common.default_conv):

        super(RCAN, self).__init__()
        
        # define head module
        self.conv = conv(n_colors, n_feats, 3)

        # define body module
        rg = lambda: ResidualGroup(conv, n_feats, 3, reduction, depth=depth)
        m = [rg() for _ in range(n_resgroups)]
        m.append(conv(n_feats, n_feats, 3))
        self.resgroups = nn.Sequential(*m)

        # define tail module
        self.recon = nn.Sequential(
            common.Upsampler(scale, n_feats, conv=conv),
            conv(n_feats, n_colors, 3),
        )

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        return {
            'scale': cfg.scale,
            'depth': cfg.depth,
            'n_colors': cfg.n_colors,
            'n_feats': cfg.n_feats,
            'n_resgroups': cfg.n_resgroups,
            'reduction': cfg.reduction,
            'conv': conv
        }

    def forward(self, x):
        x = self.conv(x)
        x = x + self.resgroups(x)
        x = self.recon(x)
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for k in own_state.keys():
            if k not in state_dict and 'recon' not in k:
                raise RuntimeError(k + ' does not exist!')
            else:
                if k in state_dict:
                    own_state[k] = state_dict[k]

        super(RCAN, self).load_state_dict(own_state, strict=strict)

