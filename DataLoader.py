import torch
import torch.nn as nn
import torchvision
from scipy import ndimage as ni
from torch.utils.data import Dataset
import pandas as pd
import os
import skimage
from skimage import io
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from MSSSIM import MS_SSIM_L1_LOSS
from torch.utils.tensorboard import SummaryWriter






class Image_Load(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return(len(self.annotations))

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        image = image.astype('float32')
        image = ni.zoom(image, [2, 2], order=1)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = self.transform(image)

        label_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        label = io.imread(label_path)
        label = label.astype('float32')
        label = (label - np.min(label)) / (np.max(label) - np.min(label))
        label = self.transform(label)

        return(image, label)

dataset = Image_Load(
    csv_file='/Users/samhexter/PycharmProjects/PHY31002/data/dataset.csv',
    root_dir='/Users/samhexter/PycharmProjects/PHY31002/data/All',
    transform=transforms.ToTensor()
)



n = 1
n_classes = 1
n_channels = 1

#Adding the Unet below

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
        return(t)


Unet = Neural_Network(n_classes, n_channels)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=n, shuffle=True)
optimizer = torch.optim.Adam(Unet.parameters(), lr=0.01)
tb = SummaryWriter()
for epoch in range(1):

    total_loss = 0



    for batch in train_loader:
        conf, ism = batch
        preds = Unet(conf)
        loss = MS_SSIM_L1_LOSS()
        output = loss(preds, ism)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        total_loss += output
        tb.add_scalar('Loss', loss)

    print(
        "epoch", epoch,
        "loss:", total_loss
    )
tb.close()
#Checking different loss mechanisms:

"""
tb = SummaryWriter()
network = Neural_Network(1,1)
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)
grid2 = torchvision.utils.make_grid(labels)
grid3 = torchvision.utils.make_grid(preds)

tb.add_image('Conf', grid)
tb.add_image('ISM', grid2)
tb.add_image('Prediction', grid3)
tb.add_scalar('Loss', loss.item())
tb.add_graph(network, images)
tb.close()
"""