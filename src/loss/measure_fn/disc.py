import torch
from torch import nn


class Discriminator(nn.Module):

    def __init__(self, ref_node):
        super().__init__()
        self.requires_ref = True

    def __str__(self):
        return 'DIS'

    def forward(self, root, ref_node):
        loss_dis = root[ref_node].f.loss
        # To prevent backpropagation
        if isinstance(loss_dis, torch.Tensor):
            loss_dis = loss_dis.item()

        return loss_dis

