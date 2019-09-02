import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.upsampling import Upsample

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.name = f"{self.__class__.__name__}"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.scaler2x = Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x):        
        x = self.scaler2x(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x