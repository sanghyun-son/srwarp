from model import common
from model.custom import stabilizer

import optimizer
from optimizer import gan_params

from misc import module_utils
from misc.gpu_utils import parallel_forward as pforward

import torch
from torch import nn
from torch import autograd
from torch.nn import utils
from torch.nn import functional


class Adversarial(nn.Module):

    def __init__(
            self,
            name: str,
            discriminator: nn.Module=None,
            _optimizer=None,
            gan_k: int=0,
            gp: float=10,
            smoothing: bool=False) -> None:

        super().__init__()
        self.name = name
        self.discriminator = discriminator
        self.optimizer = _optimizer

        self.gan_k = gan_k
        self.loss = 0
        self.loss_gradient = 0
        self.gp = 0
        self.smoothing = smoothing
        return

    @staticmethod
    def get_kwargs(cfg):
        m = module_utils.load_with_exception(cfg.dis, 'model')
        discriminator = common.get_model(m, cfg)
        '''
        Apply spectral normalization to the discriminator.
        From Miyato et al.,
        "Spectral Normalization for Generative Adversarial Networks"
        See https://arxiv.org/pdf/1802.05957.pdf for more detail.
        '''
        if not cfg.no_sn:
            for m in discriminator.modules():
                if isinstance(m, nn.modules.conv._ConvNd):
                    utils.spectral_norm(m, n_power_iterations=3)

        cfg_gan = gan_params.set_params(cfg)
        kwargs = {
            'discriminator': discriminator,
            '_optimizer': optimizer.make_optimizer(discriminator, cfg_gan),
            'gan_k': cfg.gan_k,
            'gp': cfg.gp,
            'smoothing': cfg.smoothing,
        }
        return kwargs

    def __repr__(self):
        return self.name.upper()

    def gradient_penalty(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        '''
        Calculate gradient penalty

        Args:
            fake (Tensor): a batch of fake samples
            real (Tensor): a batch of real samples

        Return:
            Tensor: gradient penalty

        Note:
            WGAN-GP - From Gulrajani et al.,
            "Improved Training of Wasserstein GANs"
            See https://arxiv.org/pdf/1704.00028.pdf for more detail.

            DRAGAN - From Kodali et al.,
            "On Convergence and Stability of GANs"
            See https://arxiv.org/pdf/1705.07215.pdf
            and https://github.com/kodalinaveen3/DRAGAN for more detail.
        '''
        # uniform random variable for interpolation
        alpha = real.new_empty(real.size(0), 1, 1, 1).uniform_()
        with torch.no_grad():
            if 'wgan' in self.name and 'gp' in self.name:
                # interpolate between real and fake
                z = real + alpha * (fake - real)
            elif 'dragan' in self.name:
                # interpolate between real and perturbed real
                real_p = real + 0.5 * real.std() * torch.rand_like(real)
                z = real + alpha * (real_p - real)

        z.requires_grad_(True)
        d_inter = pforward(self.discriminator, z)
        grad = autograd.grad(
            d_inter,
            z,
            grad_outputs=torch.ones_like(d_inter),
            create_graph=True,
        )[0]
        grad = grad.view(grad.size(0), -1)
        grad_norm = grad.norm(p=2, dim=1)
        gp = (grad_norm - 1)**2
        return gp.mean()

    def gradient_centralize(
            self,
            fake: torch.Tensor,
            real: torch.Tensor,
            d_fake: torch.Tensor,
            d_real: torch.Tensor) -> torch.Tensor:

        grad_fake = autograd.grad(
            d_fake,
            fake,
            grad_outputs=torch.ones_like(d_fake),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

        grad_real = autograd.grad(
            d_real,
            real,
            grad_outputs=torch.ones_like(d_real),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        mean_fake = grad_fake.mean(dim=-1, keepdim=True)
        mean_fake = mean_fake.mean(dim=-2, keepdim=True)

        mean_real = grad_real.mean(dim=-1, keepdim=True)
        mean_real = mean_real.mean(dim=-2, keepdim=True)

        grad_penalty = mean_fake.pow(2).mean() + mean_real.pow(2).mean()
        return grad_penalty

    def loss_d(self, fake, real):
        if 'gp' in self.name or 'gc' in self.name:
            fake.requires_grad_(True)
            real.requires_grad_(True)

        d_fake = pforward(self.discriminator, fake)
        d_real = pforward(self.discriminator, real)

        if 'w' in self.name:
            # WGAN loss
            if d_real is None:
                loss = d_fake.mean()
            else:
                loss = d_fake.mean() - d_real.mean()
        elif 'ls' in self.name:
            ls_fake = d_fake.pow(2).mean()
            ls_real = (d_real - 1).pow(2).mean()
            loss = 0.5 * (ls_fake + ls_real)
        else:
            if 'ra' in self.name:
                mean_fake = d_fake.mean()
                mean_real = d_real.mean()
                d_fake -= mean_real
                d_real -= mean_fake

            loss = self.bce(fake=d_fake, real=d_real)

        # Gradient penalty
        if 'gp' in self.name:
            self.loss_gradient = self.gradient_penalty(fake, real) 
            loss = loss + self.gp * self.loss_gradient

        if 'gc' in self.name:
            self.loss_gradient = self.gradient_centralize(
                fake, real, d_fake, d_real,
            )
            loss = loss + self.gp * self.loss_gradient

        # Update the discriminator
        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if 'w' in self.name and 'gp' not in self.name:
            # weight clipping for WGAN (not -GP)
            for m in self.modules():
                if isinstance(m, nn.modules.conv._ConvNd):
                    m.weight.data.clamp_(-0.01, 0.01)

        return loss.item()

    def loss_g(self, fake, real):
        '''
        if 'c' in self.name:
            fake = stabilizer.centralize(fake)
            real = stabilizer.centralize(real)
        '''
        d_fake = pforward(self.discriminator, fake)
        if 'w' in self.name:
            loss = -d_fake.mean()
        elif 'ls' in self.name:
            loss = (d_fake - 1).pow(2).mean()
        else:
            if 'ra' in self.name:
                with torch.no_grad():
                    d_real = pforward(self.discriminator, real)
                    mean_real = d_real.mean()

                mean_fake = d_fake.mean()
                d_fake -= mean_real
                d_real -= mean_fake
                loss = self.bce(real=d_fake, fake=d_real)
            else:
                loss = self.bce(real=d_fake)

        return loss

    def forward(self, g, z_d, real_d, fake_g, real_g):
        '''
        Args:
            g (torch.nn.Module): Generator model.
            z_d (torch.Tensor): Input of the model for updating the discriminator.
            real_d (torch.Tensor): Real samples for updating the discriminator.
            fake_g (torch.Tensor): Fake samples for updating the generator.
            real_g (torch.Tensor): Real samples for updating the generator.
        '''
        if self.training:
            self.loss = 0
            if self.gan_k == 0:
                # Use the same sample to update D and G
                self.loss = self.loss_d(fake_g.detach(), real_g)
            else:
                z_chunks = z_d.chunk(self.gan_k, dim=0)
                real_chunks = real_d.chunk(self.gan_k, dim=0)
                for z, real in zip(z_chunks, real_chunks):
                    with torch.no_grad():
                        fake = pforward(g, z)

                    self.loss += self.loss_d(fake.detach(), real)
                # Calculate the average
                self.loss /= self.gan_k
        else:
            self.loss = 0

        # For updating the generator
        loss_g = self.loss_g(fake_g, real_g)
        return loss_g

    def bce(self, fake=None, real=None):
        '''
        A binary cross entropy for valilla GANs

        Args:
            fake (Tensor, optional):
            real (Tensor):

        Return:
            Tensor: 
        '''
        if fake is None and real is None:
            raise Exception('You should provide at least one batch')

        bce_with_logits = functional.binary_cross_entropy_with_logits
        if fake is not None:
            if self.smoothing:
                zeros = torch.full_like(fake, 0.1)
            else:
                zeros = torch.zeros_like(fake)

            loss_fake = bce_with_logits(fake, zeros)
        else:
            loss_fake = 0

        if real is not None:
            if self.smoothing:
                ones = torch.full_like(real, 0.9)
            else:
                ones = torch.ones_like(real)

            loss_real = bce_with_logits(real, ones)
        else:
            loss_real = 0

        loss = loss_fake + loss_real
        return loss

