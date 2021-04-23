from model import common
from torch import nn


def model_class(cfg=None, conv=common.default_conv, make=False):
    if make:
        return SRResNet(**SRResNet.get_kwargs(cfg=cfg, conv=conv))
    else:
        return SRResNet


class SRResNet(nn.Module):
    '''
    SRResNet model

    Note:
        From Ledig et al.,
        "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
        See https://arxiv.org/pdf/1609.04802.pdf for more detail.
    '''

    def __init__(
            self,
            scale=4, depth=16, n_colors=3, n_feats=64,
            conv=common.default_conv):

        super(SRResNet, self).__init__()
        self.conv = common.BasicBlock(
            n_colors, n_feats, 9, norm=None, act='prelu',
        )

        resblock = lambda: common.ResBlock(
            n_feats, 3, norm='batch', act='prelu', conv=conv
        )
        m = [resblock() for _ in range(depth)]
        m.append(conv(n_feats, n_feats, 3))
        m.append(nn.BatchNorm2d(n_feats))
        self.resblocks = nn.Sequential(*m)
        self.recon = nn.Sequential(
            common.Upsampler(scale, n_feats, act='prelu', conv=conv),
            conv(n_feats, n_colors, 3),
        )
        self.url = None

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        return {
            'scale': cfg.scale,
            'depth': cfg.depth,
            'n_colors': cfg.n_colors,
            'n_feats': cfg.n_feats,
            'conv': conv,
        }

    def forward(self, x):
        x = self.conv(x)
        x = x + self.resblocks(x)
        x = self.recon(x)
        return x
