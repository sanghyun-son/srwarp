from trainer import base_trainer

_parent_class = base_trainer.BaseTrainer


class GANTrainer(_parent_class):

    def __init__(self, *args, gan_k=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gan_k = gan_k

    @staticmethod
    def get_kwargs(cfg):
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['gan_k'] = cfg.gan_k
        return kwargs

    def split_batch(self, **samples):
        keys = list(samples.keys())
        # Calculate the batch size
        batch_size = samples[keys[0]].size(0) // (self.gan_k + 1)
        split = self.gan_k * batch_size
        # Split samples
        samples_ret = {'real': {}, 'fake': {}}
        samples_ret['d'] = {k: v[:split] for k, v in samples.items()}
        samples_ret['g'] = {k: v[split:] for k, v in samples.items()}
        return samples_ret

    def forward(self, **samples):
        '''
        Since we are using the adversarial loss, note that
        the batch size here is (gan_k + 1) * batch_size (b).
        The first gan_k * b samples will be used to update the discriminator,
        while the remaining b samples will be used to update the generator.
        '''
        if self.training:
            samples = self.split_batch(**samples)
            z_d = samples['d']['z']
            z_g = samples['g']['z']
            real_d = samples['d']['img']
            real_g = samples['g']['img']
            # Generate a fake batch for updating the generator.
            # To save memory, we will generate fake_dis dynamically.
            # Check loss/loss_fn/adv.py for more details.
            fake_g = self.pforward(z_g)
            loss = self.loss(
                g=self.model,
                z_d=z_d,
                real_d=real_d,
                fake_g=fake_g,
                real_g=real_g,
            )
        else:
            fake_g = self.pforward(samples['z'])
            loss = self.loss(
                g=None,
                z_d=None,
                real_d=None,
                fake_g=fake_g,
                real_g=None,
            )

        return loss, {'{:0>3}'.format(self.get_epoch()): fake_g}

