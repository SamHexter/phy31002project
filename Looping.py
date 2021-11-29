import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn  # Import the neural network function from pytorch
import torch.nn.functional as F
import torch.optim as optim


torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST' #Where the data is located
    ,train=True #we want the data to be for training
    ,download=True #Download the data unless it is present at the location
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])#Transformations to be performed on the data set
)
class Network(nn.Module):
    def __init__(self):  # Constructor
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

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

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
optimizer = optim.Adam(network.parameters(), lr=0.01) #'lr'= learning rate
"""
batch = next(iter(train_loader)) #Get Batch
images, labels = batch

preds = network(images) #Pass Batch through NN
loss = F.cross_entropy(preds, labels) #Calculate Loss

loss.backward() #Calculate Gradients
optimizer.step() #Update Weights

print('Loss Initial=', loss.item())
preds = network(images)
loss = F.cross_entropy(preds,labels)
print('Loss After one loop=', loss.item())
"""
#Now we need to loop thorugh this process for one epoch


for epoch in range(1):
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        images, labels = batch

        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print("epoch;", epoch, "total_correct=", total_correct, "loss=", total_loss)
#After one epoch the NN has got 46000/60000 correct, batch_size=100 so we iterate 600 times
print(total_correct/len(train_set))

#This kept getting better and better but plateaued at ~88% accurate

#Up to video 27

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            , dim=0
        )
    return all_preds
prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
train_preds = get_all_preds(network, prediction_loader)
preds_correct = get_num_correct(train_preds, train_set.targets)

print('total correct:', preds_correct)
print('accuracy:', preds_correct / len(train_set))

stacked = torch.stack(
    (

        train_set.targets
        ,train_preds.argmax(dim=1)
    )
    ,dim=1
)

print(stacked.shape)
print(stacked)

cmt = torch.zeros(10,10, dtype=torch.int64)
#print(cmt)

for p in stacked:
    tl, pl = p.tolist() #tl = true label, pl = predicted label
    cmt[tl, pl] = cmt[tl, pl] + 1
#print(cmt)
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))

plt.figure(figsize=(10,10))

plot_confusion_matrix(cm, train_set.classes)

#plt.show()
#When the predicted label = the true label then the NN has got it right, therefore we expect the diagonal to contain the majority of the results

#Different Methods of concatonating and stacking tensors

#Concatonating joins a sequence of tensors along an existing axis
#Stacking joins a sequence of tensors along a new axis



t1_1 = torch.rand([1,1024,28,28], dtype= torch.float32)
t1_2 = torch.rand([1,512,64,64], dtype=torch.float32)

#t1 is the tensor from the bottom of the UNet that is getting upsampled
"""x1 = torch.rand([1,1024,56,56])
x2 = torch.rand([1,512,64,64])


diffY = x2.size()[2] - x1.size()[2]
diffX = x2.size()[3] - x1.size()[3]


x1 = F.pad(x1, [(diffX/2), diffX - (diffX/2),
                (diffY/2), diffY - (diffY/2)])

x = torch.cat([x2,x1], dim=1)

print(x)"""

