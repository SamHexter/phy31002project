import torch
import torch.nn as nn

class DConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        #We have in, out and mid channels as in the UNet we can modify the arms to do convolutions along different 'paths'
        super().__init__()
        #Here we use the nn.Sequential method as this class requires several operations to be made on the input tensor
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3)
            #Kernel_size=3 will perform the 3x3 convolution we need
            ,nn.BatchNorm2d(mid_channels)
            #Applies a Batch Normalization to a 4D input, this accelerates the training by reducing the internal covarient shift
            ,nn.ReLU(mid_channels)
            #Applies th rectified linear unit function element wise
            ,nn.Conv2d(mid_channels, out_channels, kernel_size=3)
            ,nn.BatchNorm2d(out_channels)
            ,nn.ReLU(out_channels)
        )
    def forward(self,t):
        t = self.dconv(t)
        return(t)
