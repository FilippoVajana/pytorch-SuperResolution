import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        identity = x        
        out = F.relu(self.conv1(x))
        out = torch.mul(self.conv2(out), 0.1)
        out = torch.add(out, identity)        
        return out


class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.residuals = self._resblocks(16)
        self.scaler2x = self._upsample2x()
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv_in(x))
        out = self.residuals(out)
        out = torch.add(out, identity)
        out = self.scaler2x(out)
        out = self.conv_out(out)
        return out

    def _resblocks(self, layers = 16):
        blocks = []
        for _ in range(layers):
            blocks.append(ResidualBlock())
        return nn.Sequential(*blocks) # * is used to unpack the list because Sequential wants args

    def _upsample2x(self):
        scaler = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        return scaler