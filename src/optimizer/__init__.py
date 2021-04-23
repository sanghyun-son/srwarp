from optimizer import adabound

import torch
from torch import optim

def make_optimizer(target, cfg):
    '''
    Make optimizer and scheduler together

    Args:

    Return:

    '''
    params = [w for w in target.parameters() if w.requires_grad]
    kwargs = {'lr': cfg.lr, 'weight_decay': cfg.weight_decay}
    if cfg.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs['momentum'] = cfg.momentum
    elif cfg.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs['betas'] = (cfg.beta1, cfg.beta2)
        kwargs['eps'] = cfg.epsilon
    elif cfg.optimizer == 'AdaBound':
        optimizer_class = adabound.AdaBound
        kwargs['final_lr'] = cfg.final_lr
        kwargs['betas'] = (cfg.beta1, cfg.beta2)
        kwargs['eps'] = cfg.epsilon
    elif cfg.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs['eps'] = cfg.epsilon
    else:
        pass

    return optimizer_class(params, **kwargs)
