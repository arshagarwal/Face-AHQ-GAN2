import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as SPN
import math

class Attention(nn.Module):
    """Attention module as per SA-GAN official implementation"""
    def __init__(self,in_channels):
        super(Attention, self).__init__()
        self.in_channels=in_channels

        self.sigma=torch.nn.parameter.Parameter(torch.tensor(0.0,requires_grad=True))

        self.maxPool=nn.MaxPool2d(kernel_size=2)

        self.theta=SPN(nn.Conv2d(self.in_channels,self.in_channels//8,1,1,bias=False))
        self.phi=SPN(nn.Conv2d(self.in_channels,self.in_channels//8,1,1,bias=False))
        self.g=SPN(nn.Conv2d(self.in_channels,self.in_channels//2,1,1,bias=False))
        self.final=SPN(nn.Conv2d(self.in_channels//2,self.in_channels,1,1,bias=False))

    def forward(self,x):
        (B,C,H,W)=x.shape
        theta = self.theta(x)
        theta = torch.reshape(theta,(theta.shape[0],theta.shape[1],-1)).permute(0,2,1) # shape (B,H*W,C/8)
        assert theta.shape == (B, H*W,self.in_channels // 8), "check theta shape Attention Module"

        phi = self.maxPool(self.phi(x))
        phi = torch.reshape(phi,(x.shape[0],self.in_channels//8,-1)) # shape(B,C/8,H*W/4)

        assert phi.shape == (B,self.in_channels//8,H*W/4), "check phi shape Attention Module phi shape :{} and Image shape is ({},{},{},{})".format(phi.shape,B,C,H,W)

        attn = torch.bmm(theta,phi)
        attn = torch.softmax(attn,dim=-1) # shape(B,H*W,H*W/4)

        g = self.maxPool(self.g(x))
        g=torch.reshape(g,(x.shape[0],self.in_channels//2,-1)) # shape=(B,C/2,H*w/4)
        g = g.permute(0,2,1) # shape=(B,H*W/4,C/2)

        attn_g = torch.bmm(attn,g).permute(0,2,1) # shape=(B,C/2,H*W)
        attn_g=torch.reshape(attn_g,(B,self.in_channels//2,H,W))
        attn_g = self.final(attn_g)

        assert attn_g.shape == x.shape,"check Attention Module"

        assert self.sigma.device==attn_g.device, "check device allocation in Attention Module"
        assert x.device==self.sigma.device, "x.device {} sigma.device {}".format(x.device,self.sigma.device)

        res=self.sigma*attn_g + x


        assert res.shape==(B,C,H,W), "check Attention Module"

        return res

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class Generator(nn.Module):
    """
    Upsamples/Downsamples Image size - 4
    """
    def __init__(self, img_size=[256,512], style_dim=64, max_conv_dim=512, w_hpf=0):
        super().__init__()
        dim_in = 2**14 // img_size[-1]
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        self.bottleneck = nn.ModuleList()


        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size[-1])) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)

            # Encoders
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))

            # Decoders
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            self.to_rgb.insert(0, nn.Sequential(nn.InstanceNorm2d(dim_in, affine=True),
                                            nn.LeakyReLU(0.2),
                                            nn.Conv2d(dim_in, 3, 1, 1, 0)))
            dim_in = dim_out


        # bottleneck blocks
        for _ in range(2):
            self.bottleneck.append(
                ResBlk(dim_out, dim_out, normalize=True))

        for _ in range(2):
            self.bottleneck.append(
                AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))



    def forward(self, x, s, img_size, alpha=0.1):
        """
        # alpha: fading parameter.
        # img_size: Integer denoting the current image size.
        """
        assert x.shape[2:] == (self.img_size[-1], self.img_size[-1]),"Check Generator input Exp: {} Got{}".format(
            (self.img_size[-1], self.img_size[-1]), x.shape
        )
        (B, C, H, W) = x.shape
        n = self.get_index(img_size)
        if img_size == self.img_size[0]:

            # Encode
            x = self.from_rgb(x)
            for block in self.encode:
                x = block(x)
            # Bottle-neck
            for block in self.bottleneck[:2]:
                x = block(x)

            for block in self.bottleneck[2:]:
                x = block(x, s)
            # Decode
            for block in self.decode[:n]:
                x = block(x, s)

            assert x.shape[2:] == (img_size, img_size), "check Zeroth Gen Got: {} Expected: {}".format(
                x.shape, (B,C,img_size, img_size))

            return self.to_rgb[n-1](x)

        else:
            # Encode
            x = self.from_rgb(x)
            for block in self.encode:
                x = block(x)

            for block in self.bottleneck[:2]:
                x = block(x)

            for block in self.bottleneck[2:]:
                x = block(x, s)

            for block in self.decode[:(n-1)]:
                x = block(x, s)
            #residual = self.to_rgb[n-2](self.temporary_upsampler(x))
            residual = self.temporary_upsampler(self.to_rgb[n-2](x))
            straight = self.to_rgb[n-1](self.decode[n-1](x, s))

            assert residual.shape == (B, C, H, W), "check residual shape in decoder Got: {} Expected: {}".format(
                                                                                    residual.shape, (B, C, H, W))
            assert straight.shape == (B, C, H, W), "check straight shape in decoder Got: {} Expected: {}".format(
                                                                                    straight.shape, (B, C, H, W))

            return (alpha * straight) + ((1 - alpha) * residual)

    def get_index(self, img_size):
        """
   #     img_size: Integer that denotes the current image size
    #    returns the number of Resnet Up/Down Sampling Blocks to be used
"""
        # number of downsampling blocks
        d = int(np.log2(self.img_size[-1])) - 4
        # number of up-sampling blocks needed for current Resolution
        u = int(np.log2(self.img_size[-1] / img_size))
        return d - u

    def temporary_upsampler(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def temporary_downsampler(self, x):
        return F.interpolate(x, scale_factor=0.5, mode='nearest')



class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    """
    Down samples Image size[0] number of times
    """
    def __init__(self, img_size=[256,512], num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size[-1]
        self.img_size = img_size
        self.blocks = nn.ModuleList()
        #blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        self.from_rgb = nn.ModuleList()

        repeat_num = int(np.log2(img_size[-1])) -2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            self.from_rgb.append(nn.Conv2d(3, dim_in, 3, 1, 1))
            dim_in = dim_out

        self.final = nn.Sequential(nn.LeakyReLU(0.2),
                                   nn.Conv2d(dim_out, dim_out, 4, 1, 0),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(dim_out, num_domains, 1, 1, 0)
                                   )
        """"
        self.blocks.append(nn.LeakyReLU(0.2))
        self.blocks.append(nn.Conv2d(dim_out, dim_out, 4, 1, 0))
        self.blocks.append(nn.LeakyReLU(0.2))
        self.blocks.append(nn.Conv2d(dim_out, num_domains, 1, 1, 0))
        """
        #self.main = nn.Sequential(*blocks)

    def forward(self, x, y, alpha=0.1, parent='Disc'):
        """
        parent: defines whether the forward is executed for generator or disc
        """
        if parent == "Gen":
            return self.gen_forward(x, y, alpha)
        else:
            img_size = x.shape[2]
            n = self.get_index(img_size)
            if img_size == self.img_size[0]:
                x = self.from_rgb[-1 * n](x)
                for block in self.blocks[(-1 * n):]:
                    x = block(x)

            else:
                straight = self.blocks[-1 * n](self.from_rgb[-1 * n](x))
                residual = self.temporary_downsampler(self.from_rgb[(-1 * n) + 1](x))
                x = (alpha * straight) + ((1 - alpha) * residual)

                for block in self.blocks[((-1*n) + 1):]:
                    x = block(x)

                out = self.final(x)

            out = self.final(x)
            out = out.view(out.size(0), -1)  # (batch, num_domains)
            idx = torch.LongTensor(range(y.size(0))).to(y.device)
            out = out[idx, y]  # (batch)
            return out

    def gen_forward(self, x, y, alpha):
        """
        returns discriminator features, Real/Fake prediction
        used in computing generator loss.
        """
        img_size = x.shape[2]
        n = self.get_index(img_size)
        if img_size == self.img_size[0]:
            x = self.from_rgb[-1 * n](x)
            for block in self.blocks[(-1 * n):]:
                x = block(x)

        else:
            straight = self.blocks[-1 * n](self.from_rgb[-1 * n](x))
            residual = self.temporary_downsampler(self.from_rgb[(-1 * n) + 1](x))
            x = (alpha * straight) + ((1 - alpha) * residual)

            for block in self.blocks[((-1 * n) + 1):]:
                x = block(x)

        return x

    def get_index(self, img_size):
        """
        img_size: Integer that denotes the current image size
        returns the number of Resnet Up/Down Sampling Blocks to be used
        """
        return int(np.log2(img_size)) - 2

    def temporary_downsampler(self, x):
        return F.interpolate(x, scale_factor=0.5, mode='nearest')
