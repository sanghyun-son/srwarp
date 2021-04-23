from data.sampler import basic_sampler
from misc import module_utils

from torch import cuda
from torch.utils import data

def get_loader(cfg, train=True):
    '''

    '''
    if train:
        dataset_list = cfg.dtrain
        # Batch multiplier when using adversarial loss
        # Check the documentation for more details
        batch_size = cfg.batch_size * (cfg.gan_k + 1)
    else:
        dataset_list = cfg.dtest
        batch_size = cfg.batch_size_eval

    def make_loader(dataset, sampler=None, train=True):
        if train and sampler is None:
            shuffle = True
        else:
            shuffle = False

        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
            num_workers=cfg.n_threads,
            pin_memory=cuda.is_available(),
        )

    dataset_dict = {}
    for name in dataset_list:
        m = module_utils.load_with_exception(name, 'data')
        data_class = module_utils.find_representative(m)
        data_kwargs = data_class.get_kwargs(cfg, train=train)
        dataset = data_class(**data_kwargs)

        if 'name' in data_kwargs:
            name = data_kwargs['name']

        dataset_dict[name] = dataset

    if train:
        dataset = data.ConcatDataset([v for v in dataset_dict.values()])
        if cfg.sampler == 'fixed':
            sampler = basic_sampler.FixedSampler(
                dataset, batch_size, cfg.test_every
            )
        elif cfg.sampler.lower() == 'none':
            sampler = None

        data_loader = make_loader(dataset, sampler=sampler, train=True)
    else:
        data_loader = {
            k: make_loader(v, train=False) for k, v in dataset_dict.items()
        }

    return data_loader
