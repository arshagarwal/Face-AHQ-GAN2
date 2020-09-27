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
from data_loader import get_loader


x = torch.randn((1,2,3))
print(x.shape[1:])