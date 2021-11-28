#Importing the practice data (Fashion MNIST)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST' #Where the data is located
    ,train=True #we want the data to be for training
    ,download=True #Download the data unless it is present at the location
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])#Transformations to be performed on the data set
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10) #We can also shuffle the data using this

#For our project the dataset won't be built in to torchvision so we will need to code it in ourselves
"""
class OHLC(Dataset):
    def _init_(self, csv_file):
        self.data = pd.read_csv(csv_file)
    def __getitem__(self, index):
        r = self.data.iloc[index]
        label = torch.tensor(r.is_up_day, dtype=torch.long)
        sample = self.normalize(torch.tensor([r.open, r.high, r.low, r.close]))
        return sample, label #Gets an item at a specific element (index) within the dataset
    def _len_(self):
        return len(self.data) #Returns the length of the dataset
"""

#Visualising Data

torch.set_printoptions(linewidth=120)

#print(train_set.train_labels.bincount())

#Each class has an equal amount of samples in each class of clothing

sample = next(iter(train_set))
image, label = sample #Sequence unpacking

plt.imshow(image.squeeze(), cmap='gray')
plt.show()
print('label:', label)

batch = next(iter(train_loader))

images, labels = batch
image_1, label_1 = train_set[0]
print(image_1.shape)
grid = torchvision.utils.make_grid(images, nrow=10)
#plt.figure(figsize=(15,15))
#plt.imshow(np.transpose(grid, (1,2,0)))
#plt.show()
#print('labels:', labels)
#This plot shows the first row of samples for training
#This can be changed in the 'batch_size' element of the train_loader code


