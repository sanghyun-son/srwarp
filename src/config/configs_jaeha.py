def add_argument(group):
    group.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
    group.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
    group.add_argument('--gen_norm', type=str, default='Instance', help='normalization layer in generator [None, Batch, Instance, Layer]')
    group.add_argument('--dis_scale', type=int, default=1, help='scale of discriminator')
    group.add_argument('--dis_norm', type=str, default='Instance', help='normalization layer in discriminator [None, Batch, Instance, Layer]')
    group.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    group.add_argument('--adl', type=bool, default=False, help='use Adaptive Data Loss')
    group.add_argument('--adl_interval', type=int, default=10, help='update interval of data loss')