import torch
from model import Discriminator

img = torch.randn((2, 3, 256, 256))
labels = torch.tensor([1, 0])
D = Discriminator(256, 2)
print(D(img, labels).shape)