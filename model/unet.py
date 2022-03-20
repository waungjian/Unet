import torch.nn as nn
import torch.nn.functional as F
import torch

class Downsample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Downsample, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,3,padding=1)
        self.conv2 = nn.Conv2d(out_channel,out_channel,3,padding=1)
        self.pool = nn.MaxPool2d(2)
    def forward(self,x,is_pool=True):
        if is_pool:
            x = self.pool(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class Upsample(nn.Module):
    def __init__(self,in_channel):
        super(Upsample, self).__init__()
        self.conv_relu = nn.Sequential(nn.Conv2d(in_channel,in_channel//2,3,padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channel//2,in_channel//2,3,padding=1),
                                       nn.ReLU())
        self.upconv =nn.ConvTranspose2d(in_channel//2,in_channel//4,3,2,padding=1,output_padding=1)

    def forward(self,input):
        x = self.conv_relu(input)
        x = F.relu(self.upconv(x))
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down1 = Downsample(3,64)
        self.down2 = Downsample(64,128)
        self.down3 = Downsample(128,256)
        self.down4 = Downsample(256,512)
        self.down5 = Downsample(512,1024)
        self.up0 = nn.ConvTranspose2d(1024,512,3,2,padding=1,output_padding=1)
        self.up1 = Upsample(1024)
        self.up2 = Upsample(512)
        self.up3 = Upsample(256)
        self.down_last = Downsample(128,64)
        self.conv_last = nn.Conv2d(64,2,1)
    def forward(self,input):
        x1 = self.down1(input,is_pool=False)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x5 = self.up0(x5)

        out = torch.cat([x4,x5],dim=1)
        out = self.up1(out)
        out = torch.cat([x3,out],dim=1)
        out = self.up2(out)
        out = torch.cat([x2,out],dim=1)
        out = self.up3(out)
        out = torch.cat([x1,out],dim=1)
        out = self.down_last(out,is_pool=False)

        return self.conv_last(out)
        


