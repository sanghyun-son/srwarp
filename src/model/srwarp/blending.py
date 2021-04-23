import typing

import torch
from torch import nn

from model.srwarp import pconv
from model.srwarp import pblock

from srwarp import transform
from srwarp import wtypes


def merge_masks(masks: typing.List[torch.Tensor]) -> torch.Tensor:
    mask_sum = sum(masks)
    mask_merge = (mask_sum == len(masks)).float()
    return mask_merge

class ContentExtractor(nn.Module):

    def __init__(
            self,
            n_pyramids: int=3,
            n_feats: int=64,
            kernel_size: int=3,
            depth: int=6) -> None:

        super().__init__()
        n_outputs = 16
        self.scale_specific = nn.ModuleList()
        for _ in range(n_pyramids):
            self.scale_specific.append(pblock.PartialResSeq(
                n_inputs=n_feats,
                n_feats=n_feats,
                n_outputs=n_outputs,
                kernel_size=kernel_size,
                depth=depth // 2,
            ))

        n_feats_ex = n_pyramids * n_outputs
        self.feature = pblock.PartialResSeq(
            n_inputs=n_feats_ex,
            n_feats=n_feats_ex,
            n_outputs=n_feats_ex,
            kernel_size=kernel_size,
            depth=(depth - depth // 2),
        )
        self.n_pyramids = n_pyramids
        self.n_outputs = n_outputs
        return

    def forward(self, ws: wtypes._LT, masks: wtypes._LT) -> torch.Tensor:
        '''
        Args:
            xs (_LT): A List of (B, C, sH, sW)
            mask (_LT): (1, 1, H, W), where mask.sum() == N

        Return:
            torch.Tensor: (B, C, H, W)
        '''
        #print('Content extractor')
        z = zip(ws, masks, self.scale_specific)
        cs = [ss(w, mask) for w, mask, ss in z]
        c = torch.cat(cs, dim=1)
        mask_merge = merge_masks(masks)
        c = c + self.feature(c, mask_merge)
        return c


class MSBlending(nn.Module):
    '''
    no_position, log_scale by default
    '''

    def __init__(
            self,
            n_pyramids: int=3,
            n_feats: int=16,
            kernel_size: int=3,
            depth: int=6) -> None:

        super().__init__()
        self.ms_feature = ContentExtractor(
            n_pyramids=n_pyramids,
            n_feats=64,
            kernel_size=kernel_size,
            depth=depth,
        )
        n_feats_ex = n_pyramids * n_feats
        m = [
            nn.Conv1d(1 + n_feats_ex, n_feats_ex, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feats_ex, n_feats_ex, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feats_ex, n_pyramids, 1),
        ]
        self.sampler = nn.Sequential(*m)
        self.n_pyramids = n_pyramids
        return

    @torch.no_grad()
    def get_s(self, j: wtypes._TT) -> torch.Tensor:
        s = transform.determinant(j)
        s = s.float()
        s = s.abs()
        s = (s + 1e-8).log()
        # Reciprocal (not that important)
        s = -s.view(1, 1, -1)
        return s

    def forward(
            self,
            ws: wtypes._LT,
            masks: wtypes._LT,
            j: wtypes._TT,
            sizes: wtypes._II,
            yi: torch.Tensor,
            **kwargs) -> torch.Tensor:
            
        '''
        Args:
            ws (typing.List[torch.Tensor]):
            masks (typing.List[torch.Tensor]):
            j (_TT):
            sizes (_II):
            yi (torch.Tensor):
            **kwargs: For handling dummy values.

        Return:
            torch.Tensor: (1, 1, sizes[0], sizes[1], n)
        '''
        c = self.ms_feature(ws, masks)
        # (B, C, N)
        c = c.view(c.size(0), c.size(1), -1)
        c = c[..., yi]

        # (1, 1, N)
        s = self.get_s(j)
        # (B, 1, N)
        s = s.repeat(c.size(0), 1, 1)

        # (B, C + 1, N)
        f = torch.cat((c, s), dim=1)
        weights_flatten = self.sampler(f)
        weights = weights_flatten.new_zeros(
            c.size(0),
            self.n_pyramids,
            sizes[0] * sizes[1],
        )
        weights[..., yi] = weights_flatten
        weights = weights.view(c.size(0), self.n_pyramids, sizes[0], sizes[1])
        z = zip(weights.split(1, dim=1), ws)
        w_blend = sum(weight * w for weight, w in z)
        return w_blend
