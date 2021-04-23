from os import path
from config import subgroup

def add_argument(group):
    with subgroup.SubGroup(group, 'general') as s:
        s.add('--dpath', type=str, default=path.join('..', 'dataset'))
        s.add('--dpath_test', type=str)
        s.add('--dtrain', nargs='+', type=str, default=['sr.div2k.base'])
        s.add('--dtest', nargs='+', type=str, default=['sr.div2k.base'])
        s.add('--train_range', type=str, default='0-100')
        s.add('--val_range', type=str, default='1-10')
        s.add('--raw', action='store_true')
        s.add('--force_ram', action='store_true')
        s.add('--sampler', type=str, default='fixed')
        s.add('--use_patch', action='store_true')

        s.add('--data_path_train', type=str)
        s.add('--data_path_test', type=str)
        s.add('--bin_path', type=str)

    with subgroup.SubGroup(group, 'property') as s:
        s.add('-s', '--scale', type=float, default=4)
        s.add('--degradation', type=str, default='bicubic')
        s.add('--degradation_test', type=str)
        s.add('--camera', type=str, default='Canon')
        s.add('--noise', type=str)
        s.add('--n_colors', type=int, default=3)

    with subgroup.SubGroup(group, 'preprocessing') as s:
        s.add('-p', '--patch', type=int, default=48)
        s.add('--augmentation', type=str, default='hvr')
        s.add('--compression', nargs='+', type=str)

    with subgroup.SubGroup(group, 'mixed') as s:
        s.add('--use_div2k', action='store_true')
        s.add('--use_ost', action='store_true')
        s.add('--use_imagenet', action='store_true')
        s.add('--use_flickr', action='store_true')
        s.add('--no_mask', action='store_true')
