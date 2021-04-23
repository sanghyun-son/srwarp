def add_argument(group):
    group.add_argument('--args', nargs='+', type=str, help='set additional arguments')
    group.add_argument('-t', '--template', default='', help='set your template')
    group.add_argument('--debug', action='store_true', help='enable debug mode')
    group.add_argument('--reproduce', type=str)

    group.add_argument('--n_threads', type=int, default=8, help='# of threads')
    group.add_argument('-g', '--gpus', type=int, default=1, help='# of GPUs')
    group.add_argument('--seed', type=int, default=2019, help='random seed')

    group.add_argument('--precision', type=str, default='single')
    group.add_argument('--sync', action='store_true')