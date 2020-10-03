import torch
import torch.nn.utils.spectral_norm as SPN

from model import Generator, Discriminator

G = Generator()
input = torch.randn((1,3,256,256))
input2 = torch.randn((1,3,512,512))
s = torch.randn((1, 64))
D = Discriminator()
print(D)
y = torch.tensor([0])
a = [2,3,5]
print(D(input2,y))