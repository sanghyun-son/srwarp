from argparse import ArgumentParser

def add_argument(group: ArgumentParser):
    group.add_argument('--transform_type', type=str, default='fixed')
    group.add_argument('--patch_max', type=int, default=1024)
    group.add_argument('--no_adaptive_down', action='store_true')
    group.add_argument('--no_adaptive_up', action='store_true')
    group.add_argument('--kernel_size_up', type=int, default=3)

    group.add_argument('--backbone', type=str, default='edsr')
    group.add_argument('--residual', action='store_true')
    group.add_argument('--kernel_net', action='store_true')
    group.add_argument('--kernel_net_multi', action='store_true')
    group.add_argument('--kernel_depthwise', action='store_true')
    group.add_argument('--kernel_bottleneck', type=int)

    group.add_argument('--depth_blending', type=int, default=6)
    group.add_argument('--depth_recon', type=int, default=10)

    group.add_argument('--adversarial', action='store_true')

    group.add_argument('--cv2_naive', action='store_true')
    group.add_argument('--cv2_interpolation', type=str, default='bicubic')

    group.add_argument('--scale_min', type=float, default=1.1)
    group.add_argument('--scale_max', type=float, default=4)

    group.add_argument('--reset_kernel', action='store_true')
    group.add_argument('--reset_sampler', action='store_true')