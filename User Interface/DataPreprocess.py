import os

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

import torch
from torchvision import transforms, utils, datasets
from PIL import Image
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(224),  # To fit popular models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507], std=[0.245]),
    # transforms.Normalize(mean=[0.507, 0.507, 0.507], std=[0.245,0.245,0.245]),
])

if __name__ == '__main__':
    train_dataset = datasets.ImageFolder(root="C:\\Users\\playc\\final_project\\train",
                                         transform=data_transform)

    image, label = train_dataset[0]
    print("--------------------------A Sample-----------------------------")
    print(image.shape)
    print(label)
    print(image, end="\n\n")

    test_dataset = datasets.ImageFolder(root="C:\\Users\\playc\\final_project\\test",
                                        transform=data_transform)

    print("--------------------------Length-----------------------------")
    print('The length of trainset:', train_dataset.__len__())
    print('The length of testset:', test_dataset.__len__())
