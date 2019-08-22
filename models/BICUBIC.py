import torch
import torch.nn as nn
import torch.nn.functional as F


class Bicubic(nn.Module):
    def __init__(self):
        super(Bicubic, self).__init__()
        self.scale_factor = 2
        self.mode = 'bicubic'

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)