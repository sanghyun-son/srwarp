from model import common
from torch import nn
from torch.nn import utils
from torch.nn import functional as F


class Discriminator(nn.Module):
    '''

    Note:
        From Ledig et al.,
        "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
        See https://arxiv.org/pdf/1609.04802.pdf for more detail.
    '''

    def __init__(
            self, depth=8, n_colors=3, width=64, norm='batch'):

        super(Discriminator, self).__init__()
        in_channels = n_colors
        out_channels = width
        stride = 1
        m = []
        for i in range(depth):
            kwargs = {'stride': stride, 'norm': norm, 'act': 'lrelu'}
            m.append(common.BasicBlock(in_channels, out_channels, 3, **kwargs))
            # reduce resolution
            stride = 2 - (i % 2)
            in_channels = out_channels
            if i % 2 == 1:
                out_channels *= 2

        self.features = nn.Sequential(*m)
        '''
        PatchGAN style
        From Isola et al.,
        "Image-to-Image Translation with Conditional Adversarial Networks"
        (pix2pix)
        See https://arxiv.org/pdf/1611.07004.pdf for more detail.
        '''
        c = []
        c.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))
        c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        c.append(nn.Conv2d(out_channels, 1, 1, padding=0, bias=False))
        self.cls = nn.Sequential(*c)

        common.init_gans(self)
        '''
        Note:
            From Miyato et al.,
            "Spectral Normalization for Generative Adversarial Networks"
            See https://arxiv.org/pdf/1802.05957.pdf for more detail.
        '''
        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                utils.spectral_norm(m, n_power_iterations=3)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        if 'wgan' in cfg.loss and 'gp' in cfg.loss:
            norm = 'layer'
        else:
            norm = 'batch'

        return {
            'depth': cfg.depth_sub,
            'width': cfg.width_sub,
            'norm': norm,
        }

    def forward(self, x):
        x = self.features(x)
        x = self.cls(x)
        return x

