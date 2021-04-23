from config import subgroup

def add_argument(group):
    '''
    From Goyal et al.,
    "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    See https://arxiv.org/pdf/1706.02677.pdf for more detail.
    '''
    group.add_argument('--linear', type=int, default=1, help='apply linear scaling')

    # learning rate scheduling
    with subgroup.SubGroup(group, 'scheduling') as s:
        s.add('--lr', type=float, default=1e-4)
        s.add('-e', '--epochs', type=int, default=300)
        s.add('--milestones', nargs='+', type=int, default=[200])
        s.add('--gamma', type=float, default=0.5)

    with subgroup.SubGroup(group, 'optimizer') as s:
        s.add(
            '--optimizer',
            type=str,
            default='ADAM',
            choices=('SGD', 'ADAM', 'AdaBound', 'RMSprop')
        )
        with subgroup.SubGroup(s.subgroup, 'SGD') as s_sgd:
            s_sgd.add('--momentum', type=float, default=0.9)

        with subgroup.SubGroup(s.subgroup, 'ADAM') as s_adam:
            s_adam.add('--beta1', type=float, default=0.9)
            s_adam.add('--beta2', type=float, default=0.999)
            s_adam.add('--epsilon', type=float, default=1e-8)

        with subgroup.SubGroup(s.subgroup, 'AdaBound') as adabound:
            adabound.add('--final_lr', type=float, default=1e-1)

    with subgroup.SubGroup(group, 'regularization') as s:
        s.add('--weight_decay', type=float, default=0)
        s.add('--grad_clip', type=float, default=0)
