from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSRB(nn.Module):

    def __init__(self, n_feats, conv=common.default_conv):
        super(MSRB, self).__init__()
        self.c1_3x3 = conv(n_feats, n_feats, 3)
        self.c2_3x3 = conv(2 * n_feats, 2 * n_feats, 3)
        self.c1_5x5 = conv(n_feats, n_feats, 5)
        self.c2_5x5 = conv(2 * n_feats, 2 * n_feats, 5)
        self.bottleneck = conv(4 * n_feats, n_feats, 1)

    def forward(self, x):
        s1 = F.relu(self.c1_3x3(x), inplace=True)
        p1 = F.relu(self.c1_5x5(x), inplace=True)
        cat = torch.cat((s1, p1), dim=1)
        s2 = F.relu(self.c2_3x3(cat), inplace=True)
        p2 = F.relu(self.c2_5x5(cat), inplace=True)
        s = self.bottleneck(torch.cat((s2, p2), dim=1))
        y = x + s

        return y

class MSRN(nn.Module):

    def __init__(self, args, conv=common.default_conv):
        super(MSRN, self).__init__()

        depth = args.depth
        n_feats = args.n_feats
        self.url = None
        self.conv0 = conv(args.n_colors, n_feats, 3)
        self.msrb_list = nn.ModuleList(
            MSRB(n_feats) for _ in range(depth)
        )
        self.bottleneck = conv((depth + 1) * n_feats, n_feats, 1)
        self.recon = nn.Sequential(
            conv(n_feats, args.scale**2 * n_feats, 3),
            nn.PixelShuffle(args.scale),
            conv(n_feats, args.n_colors, 3),
        )

    def forward(self, x):
        x = self.conv0(x)
        xs = [x]
        for msrb in self.msrb_list:
            x = msrb(x)
            xs.append(x)

        x = torch.cat(xs, dim=1)
        x = self.bottleneck(x)
        x = self.recon(x)
        return x

