import torch
import torch.nn.utils.spectral_norm as SPN

from model import Generator

G = Generator()
"""x = torch.randn((1,3,128,128))
s = torch.randn((1,64))
x = G(x,s)
torch.save(G.state_dict(),"g.ckpt")
"""
G.load_state_dict(torch.load('g.ckpt'))