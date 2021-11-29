import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn  # Import the neural network function from pytorch
import torch.nn.functional as F
import torch.optim as optim



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        #self.copy_crop1 =

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        #self.copy_crop2 =

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        #self.copy_crop3 =

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        #self.copy_crop4 =

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.conv5_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)



    def forward(self,t):
        t = t

        t = self.conv1_1(t)
        t = F.relu(t)
        t = self.conv1_2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2_1(t)
        t = F.relu(t)
        t = self.conv2_2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv3_1(t)
        t = F.relu(t)
        t = self.conv3_2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv4_1(t)
        t = F.relu(t)
        t = self.conv4_2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv5_1(t)
        t = F.relu(t)

        t = self.conv5_2(t)
        t = F.relu(t)


        return(t)
#Dconv codes the first step of the UNet where 2 convolutions occur, this takes 2 arguments, in_channels and out_channels depending on what output you would like from the convolutions
class Across(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3)
            ,nn.BatchNorm2d(mid_channels)
            ,nn.ReLU(mid_channels)
            ,nn.Conv2d(mid_channels, out_channels, kernel_size=3)
            ,nn.BatchNorm2d(out_channels)
            ,nn.ReLU(out_channels))
    def forward(self,t):
        t = self.d_conv(t)
        return(t)
#Down class Codes the downwards step and include the double convolution as described in 'Dconv'
class Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self,t):
        return(self.max_pool(t))
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    def forward(self,t1,t2):
        t1 = self.up(t1)
        diffY = t2.size()[2] - t1.size()[2]
        diffX = t2.size()[3] - t1.size()[3]

        t2 = F.pad(t2, [-diffX // 2, -diffX + diffX // 2,
                        -diffY // 2, -diffY + diffY // 2])
        t = torch.cat([t2,t1], dim=1)
        return(t)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self,t):
        t = self.conv(t)
        return(t)

class Neural_Network(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = True):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.start = Across(n_channels,64,64)
        self.Down1 = Down()
        self.Across1 = Across(64,128,128)
        self.Down2 = Down()
        self.Across2 = Across(128,256,256)
        self.Down3 = Down()
        self.Across3 = Across(256,512,512)
        self.Down4 = Down()
        self.Across4 = Across(512,1024,1024)
        self.Up1 = Up(1024,512)
        self.Across5 = Across(1024,512,512)
        self.Up2 = Up(512,256)
        self.Across6 = Across(512,256,256)
        self.Up3 = Up(256,128)
        self.Across7 = Across(256,128,128)
        self.Up4 = Up(128,64)
        self.Across8 = Across(128,64,64)
        self.end = Out(64,n_classes)
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
        return(t)

network1 = Neural_Network(1,2)
network2 = Network()

T = torch.rand([1,1,572,572], dtype=torch.float32)

print(T.shape)
print(network1(T).shape)


