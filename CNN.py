# Building the Convolutional Neural Network
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn  # Import the neural network function from pytorch
import torch.nn.functional as F

# First we need to design classes to encapsulate code and data.

# Dummy class practice

class Lizard:  # Class name
    def __init__(self, name):  # 'Self' allows us to create attribute details that get stored in the object/class
        self.name = name  # defines the class constructor

    def set_name(self, name):  # This allows us to change the name value
        # Other parameters 'name' can be stored used or saved
        self.name = name


lizard = Lizard('deep')
print(lizard.name)
# We can change this name again using set_name
lizard.set_name('lizard')
print(lizard.name)
# Attributes and methods are contained in the class/object



# First we need to extend the nn.Module base class
# Second we define layers as class attributes
# Third we implement the 'forward()' method
"""
class Network:
    def __init__(self): #Dummy layer that does nothing
        self.layer = None

    def forward(self, t):#Dummy implementation, takes tensor t and transforms it
        t = self.layer(t)
        return t
"""
# Extending upon the nn.Module
"""
class Network(nn.Module):
    def _init_(self):
        super(Network, self)._init()
        self.layer = None
    def forward(self, t):
        t = self.layer(t)
        return t
"""
# Defining layers as class attributes
"""
class Network(nn.Module):
    def _init_(self):#Constructor, needs double underscore
        super(Network, self)._init()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=(12*4*4), out_features=120)#12 comes from the number of output channels in the previous layer
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        #5 layers defined as attributes
        #2 convolutional layers
        #3 linear layers
    def forward(self, t):
    #Implement the forward pass
        t = self.layer(t)
        return t
"""


# Parameters vs Arguments
# Parameters are placeholders within a function
# Arguments are values assigned to the parameters

# Hyperparameters are parameters whose values are chosen manually and arbritrarily, mainly chosen by trial and error
# kernel_size, out_channels and out_features are hyperparameters

# What does each Parameter do?

# kernel_size sets the filter size, that will be used inside the layer
# out_channels sets the number of filters, one filter produces one output channel
# out_features sets the size of the output tensor

# Data dependant hyperparameters
# in_channels and out_features
# in_channels depend on the colour of the images
# out_features depend on the number of classes from the network

# Learnable parameters and weight tensors

# Learnable parameters are parameters whose values are learned during the
# training process. With learnable parameters, we typically start out with
# arbitrary values and then get updated in an iterative fashion until appropriate
# values that minimize the loss function appear


# We can add _repr_ to define the class representation
"""
class Network(nn.Module):
    def __init__(self):  # Constructor
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5)

        self.fc1 = nn.Linear(in_features=(12 * 4 * 4), out_features=120)  # 12 comes from the number of output channels in the previous layer
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        # 5 layers defined as attributes
        # 2 convolutional layers
        # 3 linear layers

    def forward(self, t):
        # Implement the forward pass
        t = self.layer(t)
        return t
    # def __repr__(self):
    # return "lizardnet"
"""


# how to access each layer

# print(network.conv1)#And can be done for each layer

# How to access the 'weight' inside each layer

# print(network.conv1.weight) #These weight values are updated as we train to minimize the loss function

print(network.conv1.weight.shape)  # [6,1,5,5]
# 6 represents the out_channels/filters, depth
# 1 represents the in_channels
# Both 5's represents the height and width of each filter/kernel

# All filters are represented using a single tensor
# Filters have depth that accounts for the colour channels
# print(network.fc1.weight.shape) #Gives a rank 2 tensor that encodes all of the data within the layer
# [120,192]
# 120 is from the out_features
# 192 is from the in_features
# This is from how matrix multiplication is performed

in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
weight_matrix = torch.tensor([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6]
], dtype=torch.float32)

print(weight_matrix.matmul(in_features)) #matmul is just a matrix multiplication

for param in network.parameters():
    print(param.shape)
# To access all information at once
# For each layer we get a weight tensor and bias tensor

# Creating a PyTorch linear layer

fc = nn.Linear(in_features=4, out_features=3, bias=False)



# print(fc(in_features)) #The pytorch weight matrix is filled with random values
# Adding the predetermined weight matrix gets nmuch closer to the false version
# to get the exact values we need to turn bias off in the nn.Linear function

t = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

output = fc(t)
print(output)

#Implementing the Forward Pass see line 114

#Input layer is determined by the input data
#Input layer is like the identity transformation

#Adding the input layer to the forward function
"""
def forward(self, t):
    # 1 input layer
    t = t

    # 2 Hidden Conv Layer
    t = self.conv1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # 3 Hidden conv layer
    t = self.conv2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    #Now we need to make the input layers

    # 4 hidden linear layer
    t = t.reshape(-1,12*4*4)
    t = self.fc1(t)
    t = F.relu(t)

    # 5 Hidden linear layer
    t = self.fc2(t)
    t = F.relu(t)

    # 6 Output layer
    t = self.out(t)
    #t = F.softmax(t, dim=1)# returns a positive probability for each prediction class that sum to 1
    

    return t
"""
# now we can add this to the whole CNN

class Network(nn.Module):
    def __init__(self):  # Constructor
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5)

        self.fc1 = nn.Linear(in_features=(12 * 4 * 4), out_features=120)  # 12 comes from the number of output channels in the previous layer
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        # 5 layers defined as attributes
        # 2 convolutional layers
        # 3 linear layers

    def forward(self, t):
        # 1 input layer
        t = t

        # 2 Hidden Conv Layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 3 Hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Now we need to make the input layers

        # 4 hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # 5 Hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # 6 Output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)# returns a positive probability for each prediction class that sum to 1

        return t

#Data propagates through the layers of the network

