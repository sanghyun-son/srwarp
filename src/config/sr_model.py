def add_argument(group):
    # Option for Residual dense network (RDN)
    group.add_argument('--G0', type=int, default=64,
                        help='default number of filters. (Use in RDN)')
    group.add_argument('--RDNkSize', type=int, default=3,
                        help='default kernel size. (Use in RDN)')
    group.add_argument('--RDNconfig', type=str, default='B',
                        help='parameters config of RDN. (Use in RDN)')

    # Option for Residual channel attention network (RCAN)
    group.add_argument('--n_resgroups', type=int, default=10,
                        help='number of residual groups')
    group.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')
