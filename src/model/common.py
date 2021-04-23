import random
import math
import typing

from misc import module_utils

import torch
from torch import nn
from torch.nn import functional
from torch.nn import init
from torchvision import models


def default_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int=1,
        padding: typing.Optional[int]=None,
        bias=True,
        padding_mode: str='zeros'):

    if padding is None:
        padding = (kernel_size - 1) // 2

    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        padding_mode=padding_mode,
    )
    return conv


def get_model(m, cfg, *args, make=True, conv=default_conv, **kwargs):
    '''
    Automatically find a class implementation and instantiate it.

    Args:
        m (str): Name of the module.
        cfg (Namespace): Global configurations.
        args (list): Additional arguments for the model class.
        make (bool, optional): If set to False, return model class itself.
        conv 
        kwargs (dict): Additional keyword arguments for the model class.
    '''
    model_class = module_utils.find_representative(m)
    if model_class is not None:
        if hasattr(model_class, 'get_kwargs'):
            model_kwargs = model_class.get_kwargs(cfg, conv=conv)
        else:
            model_kwargs = kwargs

        if make:
            return model_class(*args, **model_kwargs)
        else:
            return model_class
    else:
        raise NotImplementedError('The model class is not implemented!')


def model_class(model_cls, cfg=None, make=True, conv=default_conv):
    if make and hasattr(model_cls, 'get_kwargs'):
        return model_cls(**model_cls.get_kwargs(cfg, conv=conv))
    else:
        return model_cls


def init_gans(target):
    for m in target.modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()


def append_module(m, name, n_feats):
    if name is None:
        return

    if name == 'batch':
        m.append(nn.BatchNorm2d(n_feats))
    elif name == 'layer':
        m.append(nn.GroupNorm(1, n_feats))
    elif name == 'instance':
        m.append(nn.InstanceNorm2d(n_feats))

    if name == 'relu':
        m.append(nn.ReLU(True))
    elif name == 'lrelu':
        m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    elif name == 'prelu':
        m.append(nn.PReLU())


class MeanShift(nn.Conv2d):
    '''
    Re-normalize input w.r.t given mean and std.
    This module assume that input lies in between -1 ~ 1
    '''
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        '''
        Default values are ImageNet mean and std.
        '''
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        self.weight.data.copy_(torch.diag(0.5 / std).view(3, 3, 1, 1))
        self.bias.data.copy_((0.5 - mean) / std)
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    '''
    Make a basic block which consists of Conv-(Norm)-(Act).

    Args:
        in_channels (int): Conv in_channels.
        out_channels (int): Conv out_channels.
        kernel_size (int): Conv kernel_size.
        stride (int, default=1): Conv stride.
        norm (<None> or 'batch' or 'layer'): Norm function.
        act (<'relu'> or 'lrelu' or 'prelu'): Activation function.
        conv (funcion, optional): A function for making a conv layer.
    '''

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int=1,
            padding: typing.Optional[int]=None,
            norm: typing.Optional[str]=None,
            act: typing.Optional[str]='relu',
            bias: bool=None,
            padding_mode: str='zeros',
            conv=default_conv):

        if bias is None:
            bias = norm is None

        m = [conv(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )]
        append_module(m, norm, out_channels)
        append_module(m, act, out_channels)
        super().__init__(*m)


class BasicTBlock(BasicBlock):

    def __init__(self, *args, **kwargs):
        kwargs['conv'] = nn.ConvTranspose2d
        super().__init__(*args, **kwargs)


class ResBlock(nn.Sequential):
    '''
    Make a residual block which consists of Conv-(Norm)-Act-Conv-(Norm).

    Args:
        n_feats (int): Conv in/out_channels.
        kernel_size (int): Conv kernel_size.
        norm (<None> or 'batch' or 'layer'): Norm function.
        act (<'relu'> or 'lrelu' or 'prelu'): Activation function.
        res_scale (float, optional): Residual scaling.
        conv (funcion, optional): A function for making a conv layer.

    Note:
        Residual scaling:
        From Szegedy et al.,
        "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"
        See https://arxiv.org/pdf/1602.07261.pdf for more detail.

        To modify stride, change the conv function.
    '''

    def __init__(
            self,
            n_feats: int,
            kernel_size: int,
            norm: typing.Optional[str]=None,
            act: str='relu',
            res_scale: float=1,
            res_prob: float=1,
            padding_mode: str='zeros',
            conv=default_conv) -> None:

        bias = norm is None
        m = []
        for i in range(2):
            m.append(conv(
                n_feats,
                n_feats,
                kernel_size,
                bias=bias,
                padding_mode=padding_mode,
            ))
            append_module(m, norm, n_feats)
            if i == 0:
                append_module(m, act, n_feats)

        super().__init__(*m)
        self.res_scale = res_scale
        self.res_prob = res_prob
        return

    def forward(self, x):
        if self.training and random.random() > self.res_prob:
            return x

        x = x + self.res_scale * super(ResBlock, self).forward(x)
        return x


class Upsampler(nn.Sequential):
    '''
    Make an upsampling block using sub-pixel convolution
    
    Args:

    Note:
        From Shi et al.,
        "Real-Time Single Image and Video Super-Resolution
        Using an Efficient Sub-pixel Convolutional Neural Network"
        See https://arxiv.org/pdf/1609.05158.pdf for more detail
    '''

    def __init__(
            self,
            scale: int,
            n_feats: int,
            norm: typing.Optional[str]=None,
            act: typing.Optional[str]=None,
            bias: bool=True,
            padding_mode: str='zeros',
            conv=default_conv):

        bias = norm is None
        m = []
        log_scale = math.log(scale, 2)
        # check if the scale is power of 2
        if int(log_scale) == log_scale:
            for _ in range(int(log_scale)):
                m.append(conv(
                    n_feats,
                    4 * n_feats,
                    3,
                    bias=bias,
                    padding_mode=padding_mode,
                ))
                m.append(nn.PixelShuffle(2))
                append_module(m, norm, n_feats)
                append_module(m, act, n_feats)
        elif scale == 3:
            m.append(conv(
                n_feats,
                9 * n_feats,
                3,
                bias=bias,
                padding_mode=padding_mode,
            ))
            m.append(nn.PixelShuffle(3))
            append_module(m, norm, n_feats)
            append_module(m, act, n_feats)
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class UpsamplerI(nn.Module):
    '''
    Interpolation based upsampler
    '''

    def __init__(
            self, scale, n_feats, algorithm='nearest', activation=True, conv=default_conv):

        super(UpsamplerI, self).__init__()
        log_scale = int(math.log(scale, 2))
        self.algorithm = algorithm
        self.activation = activation
        self.convs = nn.ModuleList([
            conv(n_feats, n_feats, 3) for _ in range(log_scale)
        ])

    def forward(self, x):
        for conv in self.convs:
            x = functional.interpolate(x, scale_factor=2, mode=self.algorithm)
            x = conv(x)
            if self.activation:
                x = functional.leaky_relu(x, negative_slope=0.2, inplace=True)

        return x


class PixelSort(nn.Module):
    '''
    An inverse operation of nn.PixelShuffle. Only for scale 2.
    '''

    def __init__(self):
        super(PixelSort, self).__init__()

    def forward(self, x):
        '''
        Tiling input into smaller resolutions.

        Args:
            x (Tensor):

        Return:
            Tensor:

        Example::

            >>> x = torch.Tensor(16, 64, 256, 256)
            >>> ps = PixelSort()
            >>> y = ps(x)
            >>> y.size()
            torch.Size([16, 256, 128, 128])

        '''

        '''
        _, c, h, w = x.size()
        #h //= self.scale
        #w //= self.scale
        #x = x.view(-1, c, h, self.scale, w, self.scale)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        #x = x.view(-1, self.scale**2 * c, h, w)
        '''
        # we have a jit compatibility issue with the code above...
        from_zero = slice(0, None, 2)
        from_one = slice(1, None, 2)
        tl = x[..., from_zero, from_zero]
        tr = x[..., from_zero, from_one]
        bl = x[..., from_one, from_zero]
        br = x[..., from_one, from_one]
        x = torch.cat((tl, tr, bl, br), dim=1)
        return x

class Downsampler(nn.Sequential):

    def __init__(
            self, scale, n_feats,
            norm=None, act=None, conv=default_conv):

        bias = norm is None
        m = []
        log_scale = math.log(scale, 2)
        if int(log_scale) == log_scale:
            for _ in range(int(log_scale)):
                m.append(PixelSort())
                m.append(conv(4 * n_feats, n_feats, 3, bias=bias))
                append_module(m, norm, n_feats)
                append_module(m, act, n_feats)
        else:
            raise NotImplementedError

        super(Downsampler, self).__init__(*m)

def extract_vgg(name):
    gen = models.vgg19(pretrained=True).features
    vgg = None
    configs = (
        '11', '12',
        '21', '22',
        '31', '32', '33', '34',
        '41', '42', '43', '44',
        '51', '52', '53', '54',
    )
    sub_mean = MeanShift()
    def sub_vgg(config):
        sub_modules = [sub_mean]
        pool_idx = 0
        conv_idx = 0
        pools = int(config[0])
        convs = int(config[1])
        for m in gen:
            if convs == 0:
                return sub_mean
            sub_modules.append(m)
            if isinstance(m, nn.Conv2d):
                conv_idx += 1
            elif isinstance(m, nn.MaxPool2d):
                conv_idx = 0
                pool_idx += 1

            if conv_idx == convs and pool_idx == pools - 1:
                return nn.Sequential(*sub_modules)

    for config in configs:
        if config in name:
            vgg = sub_vgg(config)
            break

    if vgg is None:
        vgg = sub_vgg('54')

    return vgg

def extract_resnet(name):
    configs = ('18', '34', '50', '101', '152')
    resnet = models.resnet50
    for config in configs:
        if config in name:
            resnet = getattr(models, 'resnet{}'.format(config))
            break

    resnet = resnet(pretrained=True)
    resnet.avgpool = nn.AdaptiveAvgPool2d(1)
    resnet.fc = nn.Identity()
    resnet.eval()
    resnet_seq = nn.Sequential(MeanShift(), resnet)
    return resnet_seq


if __name__ == '__main__':
    '''
    torch.set_printoptions(precision=3, linewidth=120)
    with torch.no_grad():
        x = torch.arange(64).view(1, 1, 8, 8).float()
        ps = Downsampler(2, 1)
        print(ps(x))
        from torch import jit
        jit_traced = jit.trace(ps, x)
        print(jit_traced.graph)
        print(jit_traced)
        jit_traced.save('jit_test.pt')
        jit_load = jit.load('jit_test.pt')
        print(jit_load(x))
    '''
    x = 2 * torch.rand(1, 3, 4, 4) - 1
    print(x)
    ms = MeanShift()
    print(ms(x))
