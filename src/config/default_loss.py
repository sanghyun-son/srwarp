from config import subgroup

def add_argument(group):
    with subgroup.SubGroup(group, 'loss') as s:
        s.add('--loss', type=str, default='loss/sr_psnr.txt')
        s.add('--hparams', nargs='+', type=str, default=[])
