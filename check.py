import torch
import torch.nn.utils.spectral_norm as SPN

from model import Generator, Discriminator

a = torch.rand((1,2))
print(a.device)