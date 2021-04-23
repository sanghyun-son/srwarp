import torch
from torch import cuda
from torch.nn import parallel

def get_device(device='auto'):
    if device == 'auto':
        if cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    return device

def obj2device(x, device='auto'):
    device = get_device(device=device)
    x = x.to(device)
    return x

def dict2device(device='auto', precision='single', **x):
    device = get_device(device=device)
    to_device = lambda z: z.to(device) if isinstance(z, torch.Tensor) else z
    x = {k: to_device(v) for k, v in x.items()}
    if precision == 'half':
        def to_half(v):
            if isinstance(v, torch.Tensor):
                return v.half()
            else:
                return v

        x = {k: to_half(v) for k, v in x.items()}

    return x

def parallel_forward(model, *args, **kwargs):
    # PyTorch has a bug about this...
    device_ids = range(min(torch.cuda.device_count(), args[0].size(0)))
    return parallel.data_parallel(
        model,
        args,
        device_ids=device_ids,
        module_kwargs=kwargs,
    )
