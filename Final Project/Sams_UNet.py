import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict 


class Across(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel):
        super().__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel)
            ,nn.BatchNorm2d(mid_channels)
            ,nn.ReLU(mid_channels)
            ,nn.Conv2d(mid_channels, out_channels, kernel_size=kernel)
            ,nn.BatchNorm2d(out_channels)
            ,nn.ReLU(out_channels))
    def forward(self,t):
        t = self.d_conv(t)
        return(t)

class Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self,t):
        return(self.max_pool(t))

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    def forward(self,t1,t2):
        t1 = self.up(t1)

        diffX = t2.size()[2] - t1.size()[2]
        diffY = t2.size()[3] - t1.size()[3]

        t2 = F.pad(t2, [-diffX,0,-diffY,0])

        t = torch.cat([t2,t1], dim=1)
        return(t)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, t):
        t = self.conv(t)
        return(t)

class Final_Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(size=(1200,1200), mode='bilinear', align_corners=True)
    def forward(self, t):
        return(self.up(t))

class Neural_Network(nn.Module):
    def __init__(self, n_channels, n_classes, kernel, bilinear = True):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.start = Across(n_channels,64,64, kernel)
        self.Down1 = Down()
        self.Across1 = Across(64,128,128, kernel)
        self.Down2 = Down()
        self.Across2 = Across(128,256,256, kernel)
        self.Down3 = Down()
        self.Across3 = Across(256,512,512, kernel)
        self.Down4 = Down()
        self.Across4 = Across(512,1024,1024, kernel)
        self.Up1 = Up(1024,512)
        self.Across5 = Across(1024,512,512, kernel)
        self.Up2 = Up(512,256)
        self.Across6 = Across(512,256,256, kernel)
        self.Up3 = Up(256,128)
        self.Across7 = Across(256,128,128, kernel)
        self.Up4 = Up(128,64)
        self.Across8 = Across(128,64,64, kernel)
        self.end = Out(64,n_classes)
        self.Final = Final_Up(n_classes, n_classes)

    def forward(self,t):

        t1 = self.start(t) #This is the first tensor to be saved
        t2 = self.Across1(self.Down1(t1)) #Second tensor
        t3 = self.Across2(self.Down2(t2)) #3
        t4 = self.Across3(self.Down3(t3))#4
        t5 = self.Across4(self.Down4(t4))
        t6 = self.Up1(t5,t4)
        t = self.Across5(t6)
        t = self.Up2(t, t3)
        t = self.Across6(t)
        t = self.Up3(t, t2)
        t = self.Across7(t)
        t = self.Up4(t, t1)
        t = self.Across8(t)
        t = self.end(t)
        t = self.Final(t)

        return t
