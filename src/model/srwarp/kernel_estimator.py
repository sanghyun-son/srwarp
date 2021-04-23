import torch
from torch import nn


class KernelEstimator(nn.Sequential):

    def __init__(
            self,
            n_feats: int=48,
            kernel_size: int=3,
            depthwise: int=1) -> None:

        if depthwise == 1:
            last_layer = nn.Linear(2 * n_feats, 2 * kernel_size**2)
        else:
            last_layer = nn.Linear(2 * n_feats, 2 * depthwise * kernel_size**2)

        m = [
            nn.Linear(2 * kernel_size**2, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, 2 * n_feats),
            nn.ReLU(inplace=True),
            last_layer,
        ]
        super().__init__(*m)

        self.kernel_size = kernel_size
        self.depthwise = depthwise
        return

    def forward(self, ox: torch.Tensor, oy: torch.Tensor, threshold: float=10) -> torch.Tensor:
        ox = ox.view(ox.size(0), -1)
        oy = oy.view(oy.size(0), -1)

        '''
        # Thresholding to suppress unstable behavior
        length = ox.pow(2) + oy.pow(2)
        length.sqrt_()
        length_ok = (length < threshold).float()
        mult = length_ok + (1 - length_ok) * threshold / (length + 1e-6)
        print(mult.min(), mult.max())
        ox *= mult
        oy *= mult
        '''
        offset = torch.cat((ox, oy), dim=-1)
        k = super().forward(offset)
        kx, ky = k.chunk(2, dim=1)
        k = kx * ky
        if self.depthwise > 1:
            k = k.view(k.size(0), self.depthwise, self.kernel_size**2)
        else:
            k = k.view(k.size(0), self.kernel_size**2)

        return k