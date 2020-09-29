from model import Generator
from model import Discriminator, MappingNetwork
from data_loader import get_loader
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config):
        """Initialize configurations."""

        self.args = config

        # Model configurations.
        self.c_dim = config.c_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.num_iters = int(config.iters.split(',')[-1])
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.mode = config.mode
        self.num_workers = config.num_workers
        self.gpus = config.gpus
        self.upsample_type = config.upsample_type

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.img_dir = config.img_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # initializing progressive parameters array
        self.img_size = config.img_size.split(',')
        self.batch_size = config.batch_size.split(',')
        self.iters = config.iters.split(',')

        # converting to integer arrays
        self.img_size = [int(i) for i in self.img_size]
        self.batch_size = [int(i) for i in self.batch_size]
        self.iters = [int(i) for i in self.iters]

        print("image size is {}".format(self.img_size))

        assert len(self.img_size) == len(self.batch_size), "batch size and image size should have the same length"
        assert len(self.img_size) == len(self.iters), "batch size and iters should have the same length"

        # Build the model and tensorboard.
        self.build_model()

        # creating the data loaders array
        self.loader = self.load_data(self.img_size, self.batch_size)

    def load_data(self, img_size, batch_size):
        """
        loads the array of data loaders for corresponding batch_size and image_size
        """
        loaders = []
        for i in range(len(batch_size)):
            loaders.append(get_loader(self.img_dir, img_size[i], (img_size[i], img_size[i]), batch_size[i], mode=self.mode, num_workers=self.num_workers))

        return loaders



    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = Generator(img_size=self.img_size[0], style_dim=self.args.style_dim)
        self.D = Discriminator(img_size=self.img_size[0], num_domains=self.c_dim)
        self.M = MappingNetwork(self.args.latent_dim, self.args.style_dim, self.c_dim)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2], weight_decay=1e-4)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2], weight_decay=1e-4)
        self.m_optimizer = torch.optim.Adam(self.M.parameters(), 1e-6, [self.beta1, self.beta2], weight_decay=1e-4)

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.M, 'M')

        if self.resume_iters == None:
            print("initializing the networks")
            he_init(self.G)
            he_init(self.D)
            he_init(self.M)

        if self.gpus != "0" and torch.cuda.is_available():
            self.gpus = self.gpus.split(',')
            self.gpus = [int(i) for i in self.gpus]
            self.G = torch.nn.DataParallel(self.G, device_ids=self.gpus)
            self.D = torch.nn.DataParallel(self.D, device_ids=self.gpus)
            self.M = torch.nn.DataParallel(self.M, device_ids=self.gpus)

            
        self.G.to(self.device)
        self.D.to(self.device)
        self.M.to(self.device)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        M_path = os.path.join(self.model_save_dir, '{}-M.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.M.load_state_dict(torch.load(M_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.m_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)


    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5):
        """Generate target domain labels for debugging and testing."""

        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def get_loader_index(self, iters, count):
        """
        given iters array and count returns the index of data loader to be used
        """
        for i in range(len(iters)):
            if count <= iters[i]:
                return i
    def gen_fake(self, x_real, label_trg, load_idx):
        """
        generates fake images using progressive upsampling
        """
        x_gen = torch.nn.functional.interpolate(x_real,
                                                    scale_factor=(self.img_size[0]/self.img_size[load_idx],
                                                                  self.img_size[0]/self.img_size[load_idx]),
                                                    mode='bilinear',align_corners=True)
        assert x_gen.shape[2:] == (self.img_size[0], self.img_size[0]), "check interpolation factor in x_gen"

        z = torch.randn((x_real.size(0), self.args.latent_dim)).to(self.device)
        s_trg = self.M(z, label_trg)
        x_fake = self.G(x_gen, s_trg)
        for i in range(1, load_idx+1):
            x_fake = torch.nn.Upsample(scale_factor=2, mode=self.upsample_type)(x_fake)
            """
            Experiment later
            z = torch.randn((x_real.size(0), self.args.latent_dim)).to(self.device)
            s_trg = self.M(z, label_trg)
            """
            x_fake = self.G(x_fake, s_trg)

        return x_fake

    def train(self):
        """Progressive Training Loop."""

        loader = self.loader
        x_test = []
        y_test = []

        # Fetch fixed inputs for debugging.
        for i in range(len(loader)):
            data_iter = iter(loader[i])
            x_fixed, c_org = next(data_iter)
            x_fixed = x_fixed.to(self.device)
            #c_fixed_list = self.create_labels(c_org, self.c_dim)
            x_test.append(x_fixed)
            y_test.append(c_org)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        data = [iter(i) for i in loader]
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            load_idx = self.get_loader_index(self.iters, i+1)
            # Fetch real images and labels.
            try:
                x_real, label_org = next(data[load_idx])
            except:
                data[load_idx] = iter(loader[load_idx])
                x_real, label_org = next(data[load_idx])

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            x_real = x_real.to(self.device)           # Input images.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            x_real.requires_grad_()
            out_src = self.D(x_real, label_org)
            d_loss_real = torch.mean(torch.nn.ReLU(inplace=True)(1-out_src))
            d_loss_reg = r1_reg(out_src, x_real)

            # Compute loss with fake images.
            #x_fake = self.G(x_real, c_trg)
            x_fake = self.gen_fake(x_real, label_trg, load_idx)

            assert x_fake.shape[2:] == (self.img_size[load_idx], self.img_size[load_idx]), \
                "check fake image generation Expected: {} Got {}".format(\
                    (self.img_size[load_idx],self.img_size[load_idx]), x_fake.shape[2:])

            out_src = self.D(x_fake.detach(), label_trg)
            d_loss_fake = torch.mean(torch.nn.ReLU(inplace=True)(1+out_src))

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.args.lambda_reg*d_loss_reg
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.gen_fake(x_real, label_trg, load_idx)

                assert x_fake.shape[2:] == (self.img_size[load_idx], self.img_size[load_idx]), \
                    "check fake image generation Expected: {} Got {}".format((
                    self.img_size[load_idx], self.img_size[load_idx]), x_fake.shape[2:])

                out_src = self.D(x_fake, label_trg)
                g_loss_fake = - torch.mean(out_src)

                # Target-to-original domain.
                x_reconst = self.gen_fake(x_fake, label_org, load_idx)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                self.m_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}], Image Size {}, Batch Size {} ".format(et, i+1, self.num_iters, self.img_size[load_idx], self.batch_size[load_idx])
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    for k in range(len(loader)):
                        x_fake_list = [x_test[k]]
                        for j in range(self.c_dim):
                            label = torch.ones((x_fixed.size(0),),dtype=torch.long).to(self.device)
                            label = label*j
                            x_gen = x_test[k].clone()
                            x_fake_list.append(self.gen_fake(x_gen, label, k))
                        x_concat = torch.cat(x_fake_list, dim=3)
                        sample_path = os.path.join('samples', '{}_{}images.jpg'.format(i+1, self.img_size[k]))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                M_path = os.path.join(self.model_save_dir, '{}-M.ckpt'.format(i + 1))

                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.M.state_dict(), M_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg