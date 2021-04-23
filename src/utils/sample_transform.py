from os import path
import glob
import argparse
import types

import imageio

import torch
import tqdm

from utils import random_transform

def main(cfg: types.SimpleNamespace) -> None:
    # Crop resolution
    if cfg.test:
        imgs = sorted(glob.glob(path.join(cfg.dir, '*.png')))
        transforms = {'test': [None] * len(imgs)}
    else:
        input_size = cfg.input_size
        x = torch.zeros(1, 1, input_size, input_size)
        transforms = {
            'train': [None] * 500,
            'eval': [None] * 100,
            'test': [None] * 100,
        }

    if cfg.hard:
        scale_min = 0.2
        scale_max = 0.35
    else:
        scale_min = 0.35
        scale_max = 0.5

    for val in transforms.values():
        for i in tqdm.trange(len(val)):
            if cfg.test:
                img = imageio.imread(imgs[i])
                h, w, _ = img.shape
                if cfg.square:
                    s = min(h, w)
                    x = torch.zeros(1, 1, s, s)
                else:
                    x = torch.zeros(1, 1, h, w)

            m, _ = random_transform.get_transform(
                x,
                orientation=False,
                size_limit=1024,
                scale_min=scale_min,
                scale_max=scale_max,
                vp_min=-0.6,
                vp_max=0.6
            )
            val[i] = m

    if cfg.test:
        torch.save(transforms, path.join('utils', cfg.save_as))
    else:
        transforms['input_size'] = input_size
        torch.save(transforms, path.join('utils', cfg.save_as))

    return

if __name__ == '__main__':
    default_dir = path.join('..', 'dataset', 'DIV2K', 'DIV2K_valid_HR')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('--dir', type=str, default=default_dir)
    parser.add_argument('--save_as', type=str, default='sample.pth')
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--square', action='store_true')
    parser.add_argument('--vp_min', type=float, default=0.6)
    parser.add_argument('--vp_max', type=float, default=0.6)
    cfg = parser.parse_args()
    main(cfg)
