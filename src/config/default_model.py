from config import subgroup

def add_argument(group):
    group.add_argument('-m', '--model', type=str, default='sr.edsr')
    group.add_argument('-d', '--depth', type=int, default=16)
    group.add_argument('-f', '--n_feats', type=int, default=64)
    group.add_argument('--res_scale', type=float, default=1.0)
    group.add_argument('--res_prob', type=float, default=1.0)
    group.add_argument('--model_flag', type=str)
    group.add_argument('--normalization', type=str, default='batch')

    with subgroup.SubGroup(group, 'inheritance') as s:
        '''
        If the model is implemented based on dynamic inheritance,
        you can specify the parent model in here.
        '''
        s.add('--base', type=str, default='edsr')

    with subgroup.SubGroup(group, 'inference') as s:
        '''
        If the model has a specific inference behavior
        such as x8 forward/slice forward
        '''
        s.add('--x8', action='store_true')
        s.add('-q', '--quads', action='store_true')
