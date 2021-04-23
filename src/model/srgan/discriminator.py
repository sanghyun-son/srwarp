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
            self, depth=8, n_colors=3, width=64, patch=96, norm='batch'):

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
        if patch <= 0:
            '''
            PatchGAN style
            From Isola et al.,
            "Image-to-Image Translation with Conditional Adversarial Networks"
            (pix2pix)
            See https://arxiv.org/pdf/1611.07004.pdf for more detail.
            '''
            self.cls = nn.Conv2d(in_channels, 1, 1, padding=0)
        else:
            #Original SRGAN style
            features_resolution = patch // (2**(depth // 2))
            self.cls = nn.Sequential(
                nn.Linear(in_channels * features_resolution**2, 1024),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(1024, 1),
            )

        self.patch = patch
        common.init_gans(self)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        if 'wgan' in cfg.loss and 'gp' in cfg.loss:
            norm = 'layer'
        else:
            norm = 'batch'

        return {
            'n_colors': cfg.n_colors,
            'depth': cfg.depth_sub,
            'width': cfg.width_sub,
            'patch': cfg.dpatch,
            'norm': norm,
        }

    def forward(self, x):
        if self.patch > 0:
            # if input image sizes are not matched (only if ImageGAN)
            if x.size(-2) != self.patch or x.size(-1) != self.patch:
                return x.new_zeros(x.size(0), 1)

        x = self.features(x)
        if self.patch > 0:
            x = x.view(x.size(0), -1)

        x = self.cls(x)
        return x

