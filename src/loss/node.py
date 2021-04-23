import sys
import math
import re
import typing
import importlib

from loss import loader
from optimizer import warm_multi_step_lr

import torch
from torch import nn


class LossNode(nn.Module):

    def __init__(
            self, name, cfg, w=1, parent=None, lookup=None, arg_list=None):

        super().__init__()
        self.name = name
        self.w = float(w)
        # A simple trick to avoid infinite recursions
        self.meta_info = {'parent': parent}
        # Specify input and target
        self.arg_list = arg_list
        # Set to true if you do not want to backpropagate over this node
        self.is_measure = False
        # Parsing for children nodes
        self.parse_loss(lookup, cfg)
        # Buffers for log values
        self.log_last = 0
        self.log_avg = 0
        self.n_samples = 0
        return

    def __repr__(self):
        if self.child:
            return super().__repr__()
        else:
            target = ', '.join(self.arg_list)
            return '{:.3e} x {} ({})'.format(self.w, self.f, target)

    def __getitem__(self, key):
        if len(key) == 0:
            return self

        key_child = key.split('/')[0]
        if key_child in self.child:
            key_strip = key[len(key_child) + 1:]
            return self.child[key_child][key_strip]
        else:
            raise KeyError('Key {} does not exist!'.format(key))
            
    def get_root(self):
        parent = self.meta_info['parent']
        if parent is not None:
            return parent.get_root()
        else:
            return self

    def parse_loss(self, lookup, cfg):
        # For ordered iteration
        self.ordered_keys = []
        if self.name in lookup:
            # If current node has child(ren)
            self.child = nn.ModuleDict()
            self.f = None
            for c in lookup[self.name].split('+'):
                # Find ()
                arg_list = re.search('\(.*\)', c)
                if arg_list:
                    # Parse input and target
                    arg_list = arg_list.group(0)[1:-1]
                    arg_list = arg_list.split(',')
                    #self.arg_list = arg_list
                    # Remove ()
                    c = c.split('(')[0]
                else:
                    arg_list = self.arg_list

                parse = c.split('*')
                # Weight is 1 by default
                if len(parse) == 1:
                    w = 1
                    key = parse[0]
                else:
                    w, key = parse

                key_backup = key
                key_idx = 1
                while key in self.child:
                    key = '{}{}'.format(key_backup, key_idx)
                    key_idx += 1

                if key_idx > 1:
                    print('{} is overlapped! Assign new name: {}'.format(
                        key_backup, key,
                    ))
                self.child[key] = LossNode(
                    key_backup,
                    cfg,
                    w=w,
                    parent=self,
                    lookup=lookup,
                    arg_list=arg_list,
                )
                self.ordered_keys.append(key)
        else:
            # If current node is leaf
            self.child = {}
            self.f, self.is_measure = loader.find_module(self.name, cfg)

    def reset_log(self):
        self.log_last = 0
        self.n_samples = 0
        if self.child:
            for c in self.child.values():
                c.reset_log()

    def eval(self):
        '''
        Reset log buffers for evaluation
        '''
        self.reset_log()
        # Will automatically iterate over child
        super().eval()

    def forward(self, **kwargs):
        is_ref = hasattr(self.f, 'requires_ref') and self.f.requires_ref
        # Ignore the loss calculation when None is given
        is_none = False
        if not is_ref and not self.child:
            for k in self.arg_list:
                if k not in kwargs or kwargs[k] is None:
                    is_none = True
                elif isinstance(kwargs[k], bool) and kwargs[k] == False:
                    is_none = True

                if is_none:
                    break

        if self.w == 0 or is_none:
            loss = 0
        else:
            if self.child:
                # Iterate over children nodes
                loss = sum(v(**kwargs) for v in self.child.values())
            else:
                # Leaf node
                if is_ref:
                    '''
                    For discriminator loss,
                    we need more complicated tricks.
                    See loss/measure_fn/dis.py for more details.
                    '''
                    args = (self.get_root(), self.arg_list[0])
                else:
                    args = [kwargs[arg] for arg in self.arg_list]
                    args = [arg for arg in args if not isinstance(arg, bool)]

                enable_grad = torch.is_grad_enabled() and not self.is_measure
                # No gradient calculation for measures
                with torch.set_grad_enabled(enable_grad):
                    loss = self.f(*args)
        
        if isinstance(loss, torch.Tensor):
            log_last = loss.item()
        else:
            log_last = loss

        if loss != 0:
            self.log_last += log_last
            self.n_samples += 1

        if self.is_measure:
            # No backpropagation for measures
            return 0
        else:
            loss = self.w * loss
            return loss

    def log(self, logs=[], tag=False):
        if tag:
            logs.append(self.name)
        else:
            logs.append(self.log_last / max(1, self.n_samples))

        for k in self.ordered_keys:
            logs = self.child[k].log(logs=logs, tag=tag)

        if not tag:
            self.reset_log()

        return logs

    def write(self, writer, global_step, prefix='', postfix=''):
        '''
        Write the log to SummaryWriter
        '''
        name = self.get_name(prefix=prefix)
        if self.training:
            postfix = '/train'
        elif not postfix:
            postfix = '/eval'
        else:
            postfix = '/' + postfix

        writer.add_scalar(
            name + postfix,
            self.log_last / max(1, self.n_samples),
            global_step=global_step
        )
        # Iterate over children nodes
        if self.child:
            for c in self.child.values():
                c.write(writer, global_step)

    def register_scheduler(self, **kwargs):
        if self.child:
            for c in self.child.values():
                c.register_scheduler(**kwargs)
        else:
            if hasattr(self.f, 'optimizer'):
                self.f.scheduler = warm_multi_step_lr.WarmMultiStepLR(
                    self.f.optimizer, **kwargs
                )

    def step(self):
        if self.child:
            for c in self.child.values():
                c.step()
        else:
            if hasattr(self.f, 'scheduler') and self.w > 0:
                self.f.scheduler.step()

    def optim_state_dict(
            self,
            destination: typing.Optional[dict]=None,
            prefix: str='') -> typing.Optional[dict]:

        if destination is None:
            destination = {}

        name = self.get_name(prefix=prefix)
        # Iterate over children nodes
        if self.child:
            for c in self.child.values():
                destination = c.optim_state_dict(
                    destination=destination, prefix=name
                )
        elif hasattr(self.f, 'optimizer'):
            destination[name] = self.f.optimizer.state_dict()

        return destination

    def load_optim_state_dict(self, state_dict: str, prefix: str='') -> None:
        if self.name in state_dict:
            self.f.optimizer.load_state_dict(state_dict[self.name])

        if self.child:
            strip = lambda x: x[len(self.name) + 1:]
            state_dict_strip = {strip(k): v for k, v in state_dict.items()}
            for c in self.child.values():
                c.load_optim_state_dict(state_dict_strip)

        return

    def scheduler_state_dict(
            self,
            destination: typing.Optional[dict]=None,
            prefix: str='') -> typing.Optional[dict]:

        if destination is None:
            destination = {}

        name = self.get_name(prefix=prefix)
        # Iterate over children nodes
        if self.child:
            for c in self.child.values():
                destination = c.scheduler_state_dict(
                    destination=destination, prefix=name
                )
        elif hasattr(self.f, 'scheduler'):
            destination[name] = self.f.scheduler.state_dict()

        return destination

    def load_scheduler_state_dict(
            self,
            state_dict: str,
            prefix: str='') -> None:

        if self.name in state_dict:
            self.f.scheduler.load_state_dict(state_dict[self.name])

        if self.child:
            strip = lambda x: x[len(self.name) + 1:]
            state_dict_strip = {strip(k): v for k, v in state_dict.items()}
            for c in self.child.values():
                c.load_scheduler_state_dict(state_dict_strip)

        return

    def get_name(self, prefix=''):
        name = self.name
        if prefix:
            name = prefix + '/' + name

        return name

    def print_header(self, logger, progress=True, time=True):
        header = self.log(logs=[], tag=True)
        if progress:
            header.insert(0, 'progress')
        if time:
            header.append('time')

        header = merge_logs(header).upper()
        pretty = lambda x: '+' if x == '|' else '-'
        bar = ''.join(pretty(h) for h in header)
        logger(header)
        logger(bar)

    def print_tree(
            self, logger,
            progress=None, time_model=None, time_data=None, **kwargs):

        self.write(logger.writer, **kwargs)
        logs = self.log(logs=[])
        if progress is not None:
            logs.insert(0, '{:.0f}%'.format(progress))
        if time_model is not None and time_data is not None:
            logs.append('{:.0f}+{:.0f}'.format(time_model, time_data))

        logs = merge_logs(logs)
        logger(logs)


def merge_logs(logs, col_len=9):
    def fill_space(x):
        count = 0
        if isinstance(x, float):
            if x == 0:
                x = '0.0'
            else:
                if x < 0:
                    count += 1

                if abs(x) < 1e-4:
                    x = '{:.2g}'.format(x)
                else:
                    max_len = 4
                    log_len = math.log10(abs(x))
                    log_len = int(min(max_len, max(0, log_len)))
                    x = '{:.{}f}'.format(x, max_len - log_len)

        # For center alignment
        n_total = col_len - len(x) - 1
        n_left = math.ceil(n_total / 2)
        n_right = n_total - n_left
        x = ' ' * n_left + x + ' ' * n_right
        return x

    logs = [fill_space(log) for log in logs]
    logs = ' | '.join(logs)
    return logs
