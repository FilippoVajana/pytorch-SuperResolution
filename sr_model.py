from sr_imports import *

class Upconv(nn.Module): 
    def __init__(self, base_size):
        super(Upconv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16,kernel_size=3,stride=1,padding=1) 
        self.conv2 = nn.Conv2d(16, 32,kernel_size=3,stride=1,padding=1) 
        self.conv3 = nn.Conv2d(32, 64,kernel_size=3,stride=1,padding=1) 
        self.conv4 = nn.Conv2d(64, 128,kernel_size=3,stride=1,padding=1) 
        self.conv5 = nn.Conv2d(128, 128,kernel_size=3,stride=1,padding=1) 
        self.conv6 = nn.Conv2d(128, base_size,kernel_size=3,stride=1,padding=1) 
        self.conv7 = nn.ConvTranspose2d(base_size, 1,kernel_size=4,stride=2,padding=1,bias=False) 

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.conv7(x)
        return x


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x