from os import path
import glob
import types

import torch
from torchvision import io

from srwarp import wtypes


class SRWarpDemo(torch.utils.data.Dataset):

    def __init__(
            self,
            data_path: str=None,
            data_path_test: str=None,
            train: bool=False) -> None:

        self.gts = sorted(glob.glob(path.join(data_path, '*.png')))
        self.imgs = sorted(glob.glob(path.join(data_path_test, '*.png')))
        #self.gts = sorted(glob.glob(path.join('example', 'gt_valid', '*.png')))
        '''
        self.gts = sorted(glob.glob(path.join(
            '..',
            'dataset',
            'DIV2K',
            'DIV2K_valid_HR',
            '*.png',
        )))
        '''
        self.ms = sorted(glob.glob(path.join(data_path_test, '*.pth')))
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace, train: bool=True) -> dict:
        kwargs = {
            'data_path': cfg.dpath,
            'data_path_test': cfg.data_path_test,
            'train': train,
        }
        return kwargs

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx) -> wtypes._TT:
        gt = self.gts[idx]
        img = self.imgs[idx]
        m = self.ms[idx]

        def get_img(x: str) -> torch.Tensor:
            x = io.read_image(x)
            x = x.float()
            x = x / 127.5 - 1
            return x

        img_t = get_img(img)
        gt_t = get_img(gt)
        m = torch.load(m)
        m = m.double()

        name = path.splitext(path.basename(img))[0]
        warp_dict = {'img': img_t, 'gt': gt_t, 'm': m, 'name': name}
        return warp_dict

