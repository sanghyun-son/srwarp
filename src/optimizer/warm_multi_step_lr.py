from torch.optim import lr_scheduler

class WarmMultiStepLR(lr_scheduler.MultiStepLR):
    '''
    MultiStep learning rate scheduler with warm restart

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float, default=0.1): Multiplicative factor of learning rate decay.
        last_epoch (int, default=-1): The index of last epoch.
        
        Above descriptions are from
        https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR

        warmup (int, default=5): The number of warmup epochs.
    '''

    def __init__(
            self, optimizer, milestones,
            gamma=0.1, last_epoch=-1, linear=1, warmup=5):

        self.linear = max(linear, 1)
        self.warmup = warmup
        super().__init__(
            optimizer, milestones, gamma=gamma, last_epoch=last_epoch
        )

    def get_lr(self):
        '''
        Note: From Goyal et al.,
        "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
        See [Table 1] from https://arxiv.org/pdf/1706.02677.pdf for more detail.

        Return:
            list: a list of current learning rates.
        '''
        if self.linear == 1 or self.last_epoch > self.warmup:
            return super().get_lr()
        else:
            gradual = (self.linear - 1) / self.warmup
            scale = (1 + self.last_epoch * gradual) / self.linear
            return [scale * base_lr for base_lr in self.base_lrs]
