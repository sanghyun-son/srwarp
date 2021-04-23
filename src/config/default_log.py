from config import subgroup

def add_argument(group):
    group.add_argument('--override', type=str)
    group.add_argument('--save', type=str, default='dev')
    group.add_argument('--ablation', type=str, default='test')
    group.add_argument('--print_every', type=int, default=100)
    group.add_argument('--reset', action='store_true')

    with subgroup.SubGroup(group, 'save') as s:
        s.add('--save_period', type=int, default=10)
        s.add('--no_trace', action='store_true')
        s.add('--dump', action='store_true')
        s.add('--ext', type=str, default='png')
        s.add('--no_image', action='store_true')
