import sys
import importlib

from misc import module_utils

from torch import nn

def get_loss(m, name, cfg, **kwargs):
    '''
    Automatically find a class implementation and instantiate it.

    Args:
        m (str): Name of the module.
        cfg (Namespace): Global configurations.
        args (list): Additional arguments for the model class.
        kwargs (dict): Additional keyword arguments for the model class.
    '''
    loss_class = module_utils.find_representative(m)
    if loss_class is not None:
        if hasattr(loss_class, 'get_kwargs'):
            model_kwargs = loss_class.get_kwargs(cfg)
        else:
            model_kwargs = kwargs

        return loss_class(name, **model_kwargs)
    else:
        raise NotImplementedError('The loss class is not implemented!')

def built_in(name):
    built_in_map = {
        'l1': nn.L1Loss,
        'abs': nn.L1Loss,
        'mae': nn.L1Loss,
        'l2': nn.MSELoss,
        'mse': nn.MSELoss,
        'ce': nn.CrossEntropyLoss,
        'cls': nn.CrossEntropyLoss,
    }
    if name in built_in_map:
        return built_in_map[name]
    else:
        return None

def find_module(name, cfg, *args):
    check_built_in = built_in(name)
    if check_built_in is None:
        # String after '-' will be an argument
        name_eff = '.' + name.split('-')[0]
        loss_package = 'loss.loss_fn'
        measure_package = 'loss.measure_fn'
        is_measure = False
        try:
            # Look for the loss set first
            m = importlib.import_module(name_eff, package=loss_package)
        except:
            try:
                # Otherwise looking into the measure set
                m = importlib.import_module(name_eff, package=measure_package)
                is_measure = True
            except Exception as e:
                print(e)
                sys.exit(1)

        return get_loss(m, name, cfg), is_measure
    else:
        return check_built_in(), False

