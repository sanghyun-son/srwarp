import os
from os import path
import shutil
import sys
import math
import typing
import collections

from data import loader as dloader
from model import loader as mloader
import optimizer
from optimizer import scheduler
from loss import tree
import misc
from misc import module_utils
from misc import gpu_utils
from misc import downloader
from misc import timer
from misc import mask_utils

import torch
from torch import nn
from torch import cuda
from torch.cuda import amp
import torchvision.utils as vutils

import tqdm


def get_trainer(cfg, logger):
    m = module_utils.load_with_exception(cfg.trainer, 'trainer')
    trainer_class = module_utils.find_representative(m)
    if trainer_class is None:
        raise NotImplementedError('The trainer class is not implemented!')
    else:
        # Two arguments are dependent, so we handle them separately
        if cfg.test_only or cfg.test_period == 0:
            loader_train = None
        else:
            loader_train = dloader.get_loader(cfg, train=True)

        loader_eval = dloader.get_loader(cfg, train=False)
        network = mloader.get_model(cfg, logger=logger)
        solver = optimizer.make_optimizer(network, cfg)

        _trainer = trainer_class(
            loader_train,
            loader_eval,
            network,
            solver,
            scheduler.get_kwargs(cfg),
            tree.make_tree(cfg, logger=logger),
            misc.SRMisc(cfg),
            logger,
            **trainer_class.get_kwargs(cfg),
        )

    return _trainer


class BaseTrainer(object):

    def __init__(
            self,
            loader_train,
            loader_eval,
            network,
            solver,
            kwargs_scheduler,
            loss_tree,
            _misc,
            logger,
            grad_clip: float=0,
            resume=None,
            debug=False,
            reproduce: str=None,
            sync=False,
            precision='single',
            no_image=False):

        self.pause_count = 0
        self.logger = logger
        self.loader_train = loader_train
        self.loader_eval = loader_eval
        self.model = network
        self.optimizer = solver
        self.loss = loss_tree
        self.misc = _misc

        self.grad_clip = grad_clip
        self.debug = debug
        self.sync = sync
        self.precision = precision
        if precision == 'half':
            self.model = self.model.half()

        self.no_image = no_image

        if resume is not None:
            self.begin_epoch, self.steps = self.load_state(resume)
        else:
            self.begin_epoch = 0
            self.steps = 0

        kwargs_scheduler['last_epoch'] = self.begin_epoch - 1
        self.scheduler = scheduler.make_scheduler(
            self.optimizer,
            **kwargs_scheduler,
        )
        self.loss.register_scheduler(**kwargs_scheduler)
        self.training = True
        self.rt_log_postfix = None

        if reproduce is None:
            self.reproduce = None
        else:
            dump = torch.load(reproduce)
            self.model.load_state_dict(dump['model'])
            self.reproduce = dump

        return

    @staticmethod
    def get_kwargs(cfg):
        kwargs = {
            'grad_clip': cfg.grad_clip,
            'resume': cfg.resume,
            'debug': cfg.debug,
            'reproduce': cfg.reproduce,
            'sync': cfg.sync,
            'precision': cfg.precision,
            'no_image': cfg.no_image,
        }
        return kwargs

    def pause(self, count_max: int=0, reset: bool=False, **samples) -> None:
        '''
        This fuction is for debugging only.

        Args:
            samples (a Tensor or a tuple of Tensors):
                consists of batches of images to save.

        Note:
            We assume that -1 <= samples[i] <= 1
        '''
        print('Saving current batches...')
        save_as = path.join('..', 'debug')
        if reset and self.pause_count == 0:
            shutil.rmtree(save_as, ignore_errors=True)

        save_color_bar = False
        os.makedirs(save_as, exist_ok=True)
        for k, v in samples.items():
            normalized = None
            if isinstance(v, torch.Tensor):
                if v.dim() == 4:
                    if v.size(1) == 1 or v.size(1) == 3:
                        normalized = (v + 1) / 2
                    else:
                        normalized = None
                elif v.dim() == 3 and 'mask' in k:
                    normalized = mask_utils.code2img(v, ost=True)
                    save_color_bar = True

            if normalized is not None:
                if count_max > 0:
                    prefix = '{:0>2}_sample'.format(self.pause_count + 1)
                else:
                    prefix = 'sample'

                vutils.save_image(
                    normalized,
                    path.join(save_as, '{}_{}.png'.format(prefix, k)),
                    padding=0,
                )

        if save_color_bar:
            mask_utils.color_bar(save_as)

        self.pause_count += 1
        if self.pause_count >= count_max:
            sys.exit(0)

        return

    def get_epoch(self) -> int:
        return self.scheduler.last_epoch

    def train(self) -> None:
        '''
        Set up training environment.

        Note:
            After this call,
            - Epoch proceeds.
            - Model and loss functions will be prepared for training.
        '''
        self.training = True
        epoch = self.get_epoch() + 1
        lr = self.scheduler.get_last_lr()[0]
        self.model.train()
        self.loss.train()
        self.logger('[Epoch {}] Training (lr: {:.3e})'.format(epoch, lr))
        self.loss.print_header(self.logger)
        return

    def eval(self) -> None:
        '''
        Set up evaluation environment

        Note:
            After this call,
            - Model and loss functions will be prepared for evaluation
        '''
        self.training = False
        epoch = self.get_epoch() + 1
        self.logger('[Epoch {}] Evaluation'.format(epoch))
        self.model.eval()
        self.loss.eval()
        return

    def pforward(self, *args, **kwargs):
        '''
        Parallel forward function for multi-GPUs.
        '''
        return gpu_utils.parallel_forward(self.model, *args, **kwargs)

    def forward(self, last=True, **sample):
        '''
        Define forward behavior here.
        
        Args:
            sample (tuple): an input-target pair

        Return:
            if self.training:
                Tensor: final loss value for back-propagation
            else:
                Arbitrary types: output result(s)
        '''
        raise NotImplementedError

    def fit(self):
        time_global = timer.Timer()
        time_model = timer.Timer()
        self.train()
        self.at_epoch_begin()
        tq = tqdm.tqdm(self.loader_train, ncols=80, desc='Loss ------')
        for idx, samples in enumerate(tq):
            samples = gpu_utils.dict2device(**samples)
            # For more accurate timestamps...
            # May lower the performance
            self.synchronize()
            '''
            if self.debug:
                self.pause(**samples)
            '''
            with time_model:
                # One step for one iteration
                self.steps += 1
                loss = self.forward_with_exception(**samples)
                if isinstance(loss, torch.Tensor):
                    loss_val = loss.item()
                else:
                    loss_val = 0

                loss_str = misc.format_vp(loss_val)
                msg = 'Loss {}'.format(loss_str)
                if self.rt_log_postfix is not None:
                    msg += ' {}'.format(self.rt_log_postfix)

                tq.set_description(msg)
                
                # Update the main model
                self.optimizer.zero_grad()
                self.backward_with_exception(loss, samples)
                #self.grad_clipping()
                self.optimizer.step()
                self.synchronize()

            # these lines are for logging
            if self.misc.is_print(idx) or idx == len(self.loader_train) - 1:
                self.loss.print_tree(
                    self.logger,
                    progress=100 * (idx + 1) // len(self.loader_train),
                    time_model=time_model,
                    time_data=time_global.toc() - time_model.acc,
                    global_step=self.steps,
                )

        return

    def forward_with_exception(self, **samples) -> torch.Tensor:
        loss, _ = self.forward(**samples)
        '''
        try:
            loss, _ = self.forward(**samples)
        except Exception as e:
            print(e)
            print('Backup the current batches...')
            dump = {'model': self.model.state_dict(), 'samples': samples}
            torch.save(dump, 'debug_forward.pth')
            exit()
        '''
        return loss

    def backward_with_exception(
            self,
            loss: torch.Tensor,
            samples: typing.Mapping) -> None:

        if loss is not None and loss > 0:
            try:
                loss.backward()
            except Exception as e:
                print(e)
                print('Backup the current batches...')
                torch.save(samples, 'debug_backward.pth')
                exit()

        return

    def grad_clipping(self) -> None:
        if self.grad_clip > 0:
            params = [p for p in self.model.parameters()]
            lr = self.scheduler.get_last_lr()[0]
            c = self.grad_clip / lr
            nn.utils.clip_grad_value_(params, c)

        return

    def synchronize(self) -> None:
        if self.sync:
            cuda.synchronize()

        return

    def evaluation(self) -> None:
        t_test = timer.Timer()
        with t_test:
            self.eval()
            self.misc.join_background()
            for k in self.loader_eval.keys():
                self.evaluate_config(k)

            self.misc.end_background()

        self.logger('Total: {:.1f}s\n'.format(t_test, refresh=True))
        return

    @torch.no_grad()
    def evaluate_config(self, k):
        self.logger(k.upper())
        self.loss.reset_log()
        for idx, samples in enumerate(tqdm.tqdm(self.loader_eval[k], ncols=80)):
            if 'name' in samples:
                filename = samples['name']
            else:
                # Automatically pad zeros
                n_zeros = math.floor(math.log10(len(self.loader_eval[k])) + 1)
                filename = '{:0>{}}'.format(idx, n_zeros)

            samples = gpu_utils.dict2device(**samples, precision=self.precision)
            _, pred = self.forward(**samples)
            if not (pred is None or self.no_image):
                save_as = self.logger.get_path('res_' + k)
                self.misc.save(pred, save_as, filename[0])

        self.loss.print_header(self.logger, progress=False, time=False)
        self.loss.print_tree(
            self.logger,
            global_step=self.get_epoch(),
            postfix=k.upper(),
        )
        self.logger('', refresh=True)
        cuda.empty_cache()

    def at_epoch_begin(self) -> None:
        pass

    def at_epoch_end(self) -> None:
        # Save before .step()
        self.save_state(self.logger.get_path('latest.ckpt'))
        self.scheduler.step()
        self.loss.step()
        self.logger('', refresh=True)
        return

    def get_state(self) -> dict:
        state_model = self.model.state_dict()
        state_optimizer = self.optimizer.state_dict()
        state_scheduler = self.scheduler.state_dict()
        state_loss = self.loss.state_dict()
        state_loss_optimier = self.loss.optim_state_dict()
        state_loss_scheduler = self.loss.scheduler_state_dict()

        state_dict = {
            'epoch': self.get_epoch(),
            'steps': self.steps,
            'model': state_model,
            'optimizer': state_optimizer,
            'scheduler': state_scheduler,
            'loss': {
                'state': state_loss,
                'optimizer': state_loss_optimier,
                'scheduler': state_loss_scheduler,
            },
        }
        return state_dict

    def save_state(self, save_as: str) -> None:
        '''

        '''
        state_dict = self.get_state()
        torch.save(state_dict, save_as)
        return

    def preprocess_state(self, state: dict) -> dict:
        return state

    def load_additional_state(self, state: dict) -> None:
        return

    def load_state(self, load_from: str):
        '''

        '''
        download_prefix = 'dl-'
        model_suffix = '-model'
        if load_from.startswith(download_prefix):
            load_from = load_from.replace(download_prefix, '')
            state_dict = downloader.download(name=load_from)
            model_only = True
        else:
            if load_from.endswith(model_suffix):
                load_from = load_from.replace(model_suffix, '')
                model_only = True
            else:
                model_only = False

            print('\nResume from the checkpoint {}...'.format(load_from))
            state_dict = torch.load(load_from)

        state_model = state_dict['model']
        state_model = self.preprocess_state(state_model)

        self.load_additional_state(state_dict)
        # Why strict=False?
        # Some modules may contain pre-trained frozen parts (e.g., VGG loss).
        # We do not want to save nor load them when handling state_dicts.
        log = self.model.load_state_dict(state_model, strict=False)
        if log is not None:
            print('Missing keys:')
            print(log.missing_keys)
            print('Unexpected keys:')
            print(log.unexpected_keys)

        if model_only:
            print('Use the pre-trained model only\n')
            return 0, 0
        else:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.loss.load_state_dict(state_dict['loss']['state'], strict=False)
            self.loss.load_optim_state_dict(state_dict['loss']['optimizer'])

            last_epoch = state_dict['epoch'] + 1
            last_steps = state_dict['steps']
            return last_epoch, last_steps

