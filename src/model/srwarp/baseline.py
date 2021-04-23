import math
import types
import typing

from model.sr import mdsr_f
from model.sr import mrdb_fps
from model.sr import mrdn_f

from model.srwarp import pconv
from model.srwarp import pblock
from model.srwarp import blending
from model.srwarp import kernel_estimator

import torch
from torch import cuda
from torch import nn

from srwarp import transform
from srwarp import grid
from srwarp import warp
from srwarp import wtypes


class SuperWarpF(nn.Module):

    def __init__(
            self,
            backbone: str='mdsr',
            n_feats: int=64,
            depth_blending: int=6,
            depth_recon: int=10,
            max_scale: int=4,
            n_colors: int=3,
            residual: bool=False,
            no_adaptive_up: bool=False,
            kernel_size_up: int=3,
            kernel_net: bool=True,
            kernel_net_multi: bool=True,
            kernel_depthwise: bool=False,
            kernel_bottleneck: wtypes._I=None,
            fill: float=-255) -> None:

        super().__init__()
        if backbone == 'mdsr':
            self.backbone_multi = mdsr_f.MDSRF(
                n_feats=n_feats, bottleneck=kernel_bottleneck,
            )
        elif backbone == 'mrdb':
            self.backbone_multi = mrdb_fps.MRDBFPS(n_feats=n_feats)
        elif backbone == 'mrdn':
            self.backbone_multi = mrdn_f.MRDNF()
        else:
            raise ValueError('Invalid backbone model! ({})'.format(backbone))

        n_pyramids = int(math.log2(max_scale)) + 1
        kwargs_ms = {
            'n_pyramids': n_pyramids,
            'n_feats': 16,
            'depth': depth_blending,
        }
        kwargs_recon = {
            'n_inputs': n_feats,
            'n_feats': n_feats,
            'n_outputs': n_colors,
            'depth': depth_recon,
        }
        self.ms_blending = blending.MSBlending(**kwargs_ms)
        self.recon = pblock.PartialResRecon(**kwargs_recon)
        self.bottlenecks = None

        if kernel_net:
            if kernel_depthwise:
                if kernel_bottleneck is None:
                    depthwise = n_feats
                else:
                    depthwise = kernel_bottleneck
                    self.bottlenecks = nn.ModuleList()
                    for _ in range(n_pyramids):
                        self.bottlenecks.append(self.bottleneck_seq(
                            kernel_bottleneck, n_feats,    
                        ))
            else:
                depthwise = 1

            def make_kernel_net() -> nn.Module():
                k = kernel_estimator.KernelEstimator(
                    n_feats=n_feats,
                    kernel_size=kernel_size_up,
                    depthwise=depthwise,
                )
                return k

            if kernel_net_multi:
                self.k = nn.ModuleList()
                for _ in range(n_pyramids):
                    self.k.append(make_kernel_net())
            else:
                self.k = make_kernel_net()
        else:
            self.k = None

        self.residual = residual
        self.no_adaptive_up = no_adaptive_up
        self.kernel_size_up = kernel_size_up
        self.fill = fill
        self.pre_built = None
        self.debug = False
        self.dump = {}
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = {
            'backbone': cfg.backbone,
            'n_feats': cfg.n_feats,
            'depth_blending': cfg.depth_blending,
            'depth_recon': cfg.depth_recon,
            'max_scale': 4,
            'n_colors': cfg.n_colors,
            'residual': cfg.residual,
            'no_adaptive_up': cfg.no_adaptive_up,
            'kernel_size_up': cfg.kernel_size_up,
            'kernel_net': cfg.kernel_net,
            'kernel_net_multi': cfg.kernel_net_multi,
            'kernel_depthwise': cfg.kernel_depthwise,
            'kernel_bottleneck': cfg.kernel_bottleneck,
        }
        return kwargs

    @torch.no_grad()
    def build(self, x: torch.Tensor) -> None:
        self.pre_built = self.backbone_multi(x)
        cuda.empty_cache()
        return

    def bottleneck_seq(
            self,
            kernel_bottleneck: int,
            n_feats: int) -> nn.Module:

        m = pconv.PartialConv(kernel_bottleneck, n_feats, 1)
        return m

    def get_kernel_type(self, pi: int) -> typing.Union[str, nn.ModuleList]:
        if self.k is None:
            kernel_type = 'bicubic'
        elif isinstance(self.k, nn.ModuleList):
            kernel_type = self.k[pi]
        else:
            kernel_type = self.k

        return kernel_type

    @torch.no_grad()
    def get_global_skip(self, lr: torch.Tensor, ref: dict) -> torch.Tensor:
        if self.residual:
            ret = warp.warp_by_grid(
                lr,
                ref['grid_raw'],
                ref['yi'],
                sizes=ref['sizes'],
                kernel_type='bicubic',
                j=None,
                fill=self.fill,
            )
            return ret
        else:
            return 0

    @torch.no_grad()
    def dump_variable(self, key: str, value: torch.Tensor) -> None:
        if self.debug:
            self.dump[key] = value.detach().clone()

        return

    def forward(
            self,
            lr: torch.Tensor,
            m: typing.Union[torch.Tensor, typing.Callable],
            sizes: typing.Optional[wtypes._II]=None,
            exact: bool=False,
            debug: bool=False) -> wtypes._TT:

        '''
        Args:
            lr (torch.Tensor): An input LR image.
            m (torch.Tensor): A transformation matrix or function.
            sizes (_II): The target image size.
            debug (bool): Get intermediate results for debugging.
        '''
        self.debug = debug

        if isinstance(m, torch.Tensor):
            m = m.cpu().double()

        if sizes is None:
            sizes = (lr.size(-2), lr.size(-1))

        # [(B, C, h, w), (B, C, 2H, 2W), (B, C, 4H, 4W)]
        pyramids = self.backbone_multi(lr)
        ws = []
        masks = []
        ref = {}
        for pi, p in enumerate(pyramids):
            s = lr.size(-1) / p.size(-1)
            sizes_source = (p.size(-2), p.size(-1))
            if isinstance(m, torch.Tensor):
                with torch.no_grad():
                    ms = transform.compensate_scale(m, s)
                    # Backup these values to keep consistency between scales
                    if pi == 0:
                        ms, sizes_1, offsets_1 = transform.compensate_matrix(
                            p, ms, exact=exact, debug=False,
                        )
                        ref['sizes'] = sizes_1
                        ref['offsets'] = offsets_1
                    else:
                        dy, dx = ref['offsets']
                        ms = transform.compensate_offset(
                            ms, dx, dy, offset_first=False,
                        )

                    ms_inv = transform.inverse_3x3(ms)

                grid_raw, yi = grid.get_safe_projective_grid(
                    ms_inv, ref['sizes'], sizes_source,
                )
                j = transform.jacobian(ms_inv, sizes=ref['sizes'], yi=yi)
            else:
                if pi == 0:
                    ref['sizes'] = sizes
                    ref['offsets'] = (0, 0)

                grid_raw, yi = grid.get_safe_functional_grid(
                    m, ref['sizes'], sizes_source, scale=s,
                )
                j = transform.jacobian(m, sizes=ref['sizes'], yi=yi, scale=s)

            kernel_type = self.get_kernel_type(pi)
            if self.no_adaptive_up:
                j_warp = None
            else:
                j_warp = j

            w = warp.warp_by_grid(
                p,
                grid_raw,
                yi,
                sizes=ref['sizes'],
                kernel_type=kernel_type,
                j=j_warp,
                regularize=(kernel_type == 'bicubic'),
                fill=self.fill
            )
            with torch.no_grad():
                mask = (w[:1, :1] != self.fill).float()

            if self.bottlenecks is not None:
                w = self.bottlenecks[pi](w, mask)

            ws.append(w)
            masks.append(mask)

            # For debugging
            self.dump_variable('w_{}'.format(pi), w)
            self.dump_variable('mask_{}'.format(pi), mask)

            # Backup these values to keep consistency between scales
            if pi == 0:
                ref['grid_raw'] = grid_raw
                ref['yi'] = yi
                ref['j'] = j

        sr = self.ms_blending(ws, masks, **ref)
        sr = self.recon(sr, masks[0])

        sr = sr + self.get_global_skip(lr, ref)
        mask_merge = blending.merge_masks(masks)
        sr = mask_merge * sr + (1 - mask_merge) * self.fill
        self.dump_variable('sr', sr[0:1])

        dy, dx = ref['offsets']
        sr_full = sr.new_full((sr.size(0), sr.size(1), *sizes), self.fill)
        sr_full[..., -dy:(sr.size(-2) - dy), -dx:(sr.size(-1) - dx)] = sr
        mask_full = (sr_full != self.fill).float()
        return sr_full, mask_full
