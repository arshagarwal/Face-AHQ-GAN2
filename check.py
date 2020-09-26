import torch
import torch.nn.utils.spectral_norm as SPN
from model import ResidualBlock,Attention,Generator
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


def get_loader(image_dir,crop_size=178, image_size=128,
               batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

data1 = get_loader('slim_dataset/Train_dataset', 128, (128,128), 3)
data2 = get_loader('slim_dataset/Train_dataset', 128, (256,256), 3)
data1= iter(data1)
x = next(data1)
print(len(x))
print(x[0].shape)
data2= iter(data2)
x = next(data2)
print(len(x))
print(x[0].shape)

