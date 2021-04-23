from trainer.gan import dcgan
from misc.gpu_utils import parallel_forward as pforward
from model.utils import forward_utils as futils
_parent_class = dcgan.GANTrainer


class SRTrainer(_parent_class):

    def __init__(self, *args, x8=False, quads=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.x8 = x8
        self.quads = quads

    @staticmethod
    def get_kwargs(cfg):
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['x8'] = cfg.x8
        kwargs['quads'] = cfg.quads
        return kwargs

    def forward(self, **samples):
        if self.training:
            samples = self.split_batch(**samples)
            lr_d = samples['d']['lr']
            lr_g = samples['g']['lr']
            hr_d = samples['d']['hr']
            hr = samples['g']['hr']

            sr = pforward(self.model, lr_g)
            loss = self.loss(
                lr=lr_g,
                g=self.model,
                lr_d=lr_d,
                hr_d=hr_d,
                sr=sr,
                hr=hr,
            )
        else:
            lr = samples['lr']
            if self.quads:
                sr = futils.quad_forward(self.model, lr)
            elif self.x8:
                sr = futils.x8_forward(self.model, lr)
            else:
                sr = pforward(self.model, lr)

            loss = self.loss(
                lr=lr,
                g=None,
                lr_d=None,
                hr_d=None,
                sr=sr,
                hr=samples['hr'],
            )

        return loss, sr
