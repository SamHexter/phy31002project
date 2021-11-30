import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import math

from collections import OrderedDict

torch.set_printoptions(linewidth=150)

#Instead of using the def forward method we can use this sequential method to do it in one function

train_set = torchvision.datasets.FashionMNIST(
    root = './data'
    ,train=True
    ,download=True
    ,transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

image, label = train_set[0]
#print(image.shape)

#plt.imshow(image.squeeze(), cmap='gray')
#plt.show()

in_features = image.numel()
print(in_features)
out_features = math.floor(in_features/2)
print(out_features)
out_classes = len(train_set.classes)

network1 = nn.Sequential(
    nn.Flatten(start_dim=1)
    ,nn.Linear(in_features, out_features)
    ,nn.Linear(out_features, out_classes)
)
print(network1)
image = image.unsqueeze(0)
print(image.shape)

layers = OrderedDict([
    ('flat', nn.Flatten(start_dim=1))
    ,('hidden', nn.Linear(in_features, out_features))
    ,('output', nn.Linear(out_features, out_classes))
])

network2 = nn.Sequential(layers)
print(network2)

torch.manual_seed(50)
network3 = nn.Sequential()
network3.add_module('flat', nn.Flatten(start_dim=1))

