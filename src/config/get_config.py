import os
from os import path
import random
import argparse

from misc import module_utils
from config import template

import torch
import numpy as np


def parse():
    default_parse = [
        'default_general',
        'default_data',
        'default_model',
        'default_loss',
        'default_trainer',
        'default_optimizer',
        'default_log',
        'default_gans',
        'sr_model',
        'cfg_srwarp',
    ]
    parser = argparse.ArgumentParser()
    for option in default_parse:
        group = parser.add_argument_group(option)
        m = module_utils.load_with_exception(option, 'config')
        m.add_argument(group)

    cfg = parser.parse_args()
    template.set_template(cfg.template, cfg)

    if cfg.override is not None:
        # --args1 [value1] -args2 [value2] ...
        args = cfg.override.split('--')[1:]
        for kv in args:
            kv_split = kv.split(' ')
            k = kv_split[0]
            v = kv_split[1]
            if v.isdecimal():
                v = int(v)
            elif v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False

            print('{}: {} is overrided to {}'.format(k, getattr(cfg, k), v))
            setattr(cfg, k, v)

    # Resume from the latest checkpoint
    if cfg.resume == 'self':
        cfg.resume = path.join('..', 'experiment', cfg.save, 'latest.ckpt')
        cfg.reset = False

    if cfg.seed == -1:
        cfg.seed = random.randint(0, 2**31 - 1)

    split = cfg.save.split(os.sep)
    if len(split) == 2:
        cfg.save, cfg.ablation = split

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.linear > 1:
        cfg.batch_size *= cfg.linear
        cfg.lr *= cfg.linear
        cfg.print_every = max(cfg.print_every // cfg.linear, 1)
        cfg.test_every = max(cfg.test_every // cfg.linear, 1)

    return cfg

def parse_namespace(cfg, *args):
    '''
    Get *args from cfg.

    Args:
        cfg (napespace): Target namespace.
        *args (list of str): Argument names to be parsed.

    Example::
        >>> parse_namespace(test, 'a', 'b', 'c')
        >>> {'a': cfg.a, 'b': cfg.b, 'c': cfg.c}
    '''
    ret = {}
    for arg in args:
        if hasattr(cfg, arg):
            ret[arg] = getattr(cfg, arg)

    return ret

