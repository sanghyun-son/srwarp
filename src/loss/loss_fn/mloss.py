from torch import nn


class MaskedDist(nn.Module):
    '''
    Masked L1/L2 loss between images.
    '''
    def __init__(self, name: str) -> None:
        super().__init__()
        self.l2 = 'l2' in name
        
    def forward(self, x, y, mask):
        diff = mask * (x - y)
        if self.l2:
            diff = diff.pow(2)
        else:
            diff = diff.abs()

        gain = mask.nelement() / mask.sum()
        loss = gain.item() * diff.mean()
        return loss


if __name__ == '__main__':
    pass
