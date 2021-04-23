from optimizer import warm_multi_step_lr

def get_kwargs(cfg):
    return {
        'milestones': cfg.milestones,
        'gamma': cfg.gamma,
        'linear': cfg.linear,
    }

def make_scheduler(opt, **kwargs):
    return warm_multi_step_lr.WarmMultiStepLR(opt, **kwargs)
