from os import path
import math
import re

from loss import node
from misc import gpu_utils

import torch
from torch import nn

def make_tree(cfg, logger=None):
    '''
    Make loss tree

    Args:
        cfg (namespace): Configuration file directory
        hparams (list, optional): Hyperparameters to be overwritten

    Example::
        make_tree(cfg, ['w1=1', 'w2=2'])
        w1 and w2 in the cfg file will be replaced to 1 and 2, respectively.
    '''
    if logger is not None:
        logger('\n[Preparing loss...]')

    loss_file = cfg.loss
    if not loss_file.lower().endswith('.txt'):
        loss_file += '.txt'

    with open(loss_file, 'r') as f:
        lines = f.read().splitlines()

    lines = parse(lines)
    hparams = parse(cfg.hparams)

    # Overwrite hyperparameters
    for k, v in hparams.items():
        if k in lines:
            lines[k] = v

    # Replace variables
    for k, v in lines.items():
        meta_k = '$({})'.format(k)
        for kk, vv in lines.items():
            if meta_k in vv:
                lines[kk] = vv.replace(meta_k, v)

    # Root node
    root = node.LossNode('total', cfg, lookup=lines)

    # Print the total graph
    if logger is not None:
        logger(root)
        with open(logger.get_path('loss.txt'), 'w') as f:
            for k, v in lines.items():
                f.write('{}={}\n'.format(k, v))

    # Use GPU
    root = gpu_utils.obj2device(root)
    return root

def parse(lines):
    '''
    Parsing function for preprocessing

    Args:
        lines (list): List of lines (str).
    '''
    # Remove empty lines
    lines = [l for l in lines if l]
    # Remove comments
    lines = [l for l in lines if '#' not in l]
    # Remove spaces
    lines = [l.replace(' ', '') for l in lines]
    # Join splited lines
    idx = 1
    while idx < len(lines):
        spliter = '\\'
        if lines[idx - 1][-1] == spliter:
            pop = lines.pop(idx)
            new_line = lines[idx - 1].replace(spliter, '') + pop
            lines[idx - 1] = new_line
        else:
            idx += 1

    '''
    # Make function signiture
    funcs = {}
    for line in lines:
        if '@' in line:
            primitive = line.split('=')
            if len(primitive) != 2:
                raise ValueError(
                    'Function should be expressed in a form: A(@) = B(@)'
                )

            name, expression = primitive
            name = name.split('(')[0]
            args = re.search('\(.*\)', line)
            if args:
                arg_list = args.group(0)[1:-1]
                arg_list = args.split(',')

            funcs[name] = {'args': arg_list, 'expression': expression}
    '''
    # Remove functions from the main script
    lines = [l for l in lines if '@' not in l]

    lines = [l.split('=') for l in lines]
    lines = {k: v for k, v in lines}
    return lines
