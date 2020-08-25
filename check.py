import torch
import torch.nn.utils.spectral_norm as SPN
from model import ResidualBlock,Attention,Generator

a=torch.nn.Conv2d(3,64,4,2,bias=False)

inp = torch.randn((1,3,128,128))
c=torch.tensor([[1.,0.]])
G=Generator(c_dim=2)
print(G(inp,c))