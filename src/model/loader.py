from misc import gpu_utils
from misc import module_utils

def get_model(cfg, logger=None):
    if logger is not None:
        logger('\n[Preparing model...]')

    m = module_utils.load_with_exception(cfg.model, 'model')
    model_class = module_utils.find_representative(m)
    if model_class is None:
        _model = m.make_model(cfg)
    else:
        model_kwargs = model_class.get_kwargs(cfg)
        _model = model_class(**model_kwargs)

    n_params = sum(p.nelement() for p in _model.parameters())
    if logger is not None:
        logger(_model, clip=10, filename='model.txt')
        logger('# Parameters: {:,}'.format(n_params))

    _model = gpu_utils.obj2device(_model)
    return _model
