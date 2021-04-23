import glob
from os import path
import random
import argparse
import typing

import torch
from torchvision import io
import tqdm

from srwarp import transform

@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_as', type=str, default='transform.pth')
    parser.add_argument('--patch_size', type=int, default=384)
    parser.add_argument('--margin', type=int, default=20)
    cfg = parser.parse_args()

    div2k = path.join(path.expanduser('~'), 'dataset', 'DIV2K')
    meta = {
        'train': {
            'path': path.join(div2k, 'DIV2K_train_HR', '*.png'),
            'n': 10,
            'crop': cfg.patch_size,
        },
        'eval': {
            'path': path.join(div2k, 'DIV2K_valid_HR', '*.png'),
            'n': 1,
            'crop': cfg.patch_size,
        },
        'test': {
            'path': path.join(div2k, 'DIV2K_valid_HR', '*.png'),
            'n': 1,
            'crop': None,
        }
    }
    transforms = {k: [] for k in meta.keys()}
    transforms['patch_size'] = cfg.patch_size

    for k, v in meta.items():
        imgs = sorted(glob.glob(v['path']))
        for img in tqdm.tqdm(imgs, ncols=80):
            x = io.read_image(img)
            for _ in range(v['n']):
                m = transform.get_random_transform(
                    x, size_limit=1024, max_iters=50,
                )
                h = x.size(-2)
                w = x.size(-1)
                if v['crop'] is not None:
                    py = random.randrange(0, h - v['crop'] - 2 * cfg.margin + 1)
                    px = random.randrange(0, w - v['crop'] - 2 * cfg.margin + 1)
                    x_crop = x[..., py:(py + v['crop']), px:(px + v['crop'])]
                    m = transform.compensate_offset(m, px, py)
                    m, _, _ = transform.compensate_matrix(x_crop, m)

                transforms[k].append(m)

    torch.save(transforms, cfg.save_as)
    return

if __name__ == '__main__':
    main()