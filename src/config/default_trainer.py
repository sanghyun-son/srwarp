def add_argument(group):
    group.add_argument('--trainer', type=str, default='sr.base')
    group.add_argument('--test_every', type=int, default=1000)
    group.add_argument('--resume', type=str)
    group.add_argument('--batch_size', type=int, default=16)
    group.add_argument('--batch_size_eval', type=int, default=1)
    group.add_argument('--test_period', type=int, default=1)
    group.add_argument('--test_only', action='store_true')
    group.add_argument('--amp', action='store_true')
