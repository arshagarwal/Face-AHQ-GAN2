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
import torch.nn.functional


x = torch.randn((1,3,128,128))
print(torch.nn.functional.interpolate(x, scale_factor=(0.25, 0.25), mode='bilinear',align_corners=True).shape)