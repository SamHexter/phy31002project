# My attempt at converting the data loader into a class

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import os
import random


class DL(nn.Module):
    def __init__(self, dname1, dname2):
        self.dir1 = dname1
        self.dir2 = dname2

    # I'm going to put the loadtiffs function in here
    @classmethod
    def loadtiffs(cls, img):
        imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
        for I in range(img.n_frames):
            img.seek(I)
            imgArray[:, :, I] = np.asarray(img)
        img.close()
        return (imgArray)

    @classmethod
    def Load(cls, dname, n):
        img_array = []

        for fname in os.listdir(dname):
            im = Image.open(os.path.join(dname, fname))  # finds all the tiff files in the specific directory
            img_array.append(im)

        # removes unwanted data
        if dname == '/content/drive/Shareddrives/Team Net/Training Data/Conf_Train':
            img_array.pop(19)
            img_array.pop(19)
            img_array.pop(19)
            img_array.pop(19)
            img_array.pop(30)
            img_array.pop(30)
            img_array.pop(30)

        # Now to randomise the array
        # Create an list from 1 to n
        indexes = []
        for i in range(0, len(img_array)):
            indexes.append(i)

        # create a random list from the indexes
        rand_indexes = random.sample(indexes, len(indexes))

        # randomising images using the random indexes
        img_rand_array = []

        for i in range(len(img_array)):
            m = rand_indexes[i]
            img_rand_array.append(img_array[m])

        # the PIL images are ran through the loadtiffs function
        array = []
        for i in range(len(img_rand_array)):
            data = cls.loadtiffs(img_rand_array[i])
            array.append(data)

        # moves the axis to fit a tensor shape
        Array = np.asarray(array)
        Array = np.moveaxis(Array, -1, 1)

        # inputting batch
        Batch = []
        for i in range(0, n):
            Batch.append(Array[i])
        Batch = np.asarray(Batch)
        return Batch


# Call an instance of the class
# As the @classmethod has been used the function within the class 'Load' can be called
# Ive got it in a class with no errors but it wont let me call th function

dname1 = '/content/drive/Shareddrives/Team Net/Training Data/Conf_Train'
dname2 = '/content/drive/Shareddrives/Team Net/Training Data/ISM_Train'

Class1 = DL.Load(dname1, 2)
Class2 = DL.Load(dname2, 2)

# Looking promising just need to work out th kinks
# I'm not sure why we need to input a variable for loadtiffs when it gets called inside 'Load' and thats where the variable is defined

print(Class1.shape)
print(Class2.shape)

# It works!!!