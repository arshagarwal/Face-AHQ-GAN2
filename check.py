import torch
import torch.nn.utils.spectral_norm as SPN

from model import Generator, Discriminator

G = Generator()
input = torch.randn((1,3,256,256))
input2 = torch.randn((1,3,512,512))
s = torch.randn((1, 64))
D = Discriminator()
y = torch.tensor([0])
a = 5
a = torch.tensor(a)
print(a)