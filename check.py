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


input=torch.randn((1,3,128,128))
c = torch.tensor([[1.,0]])
G=Generator(c_dim=2)
x=G(input, c)
x=G(x, c)
print(x)