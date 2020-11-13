import torch
import torch.nn.utils.spectral_norm as SPN
import os
from model import Generator, Discriminator


def test(self):
    loader = self.loader
    loader = [iter(i) for i in loader]

    # Translate fixed images for debugging.
    with torch.no_grad():
        for k in range(len(loader)):
            dir_path = self.result_dir + "/{}_resolution".format(self.img_size[k])
            os.mkdir(dir_path)
            # x_fake_list = []
            try:
                x_test, _ = next(loader[k])
            except:
                x_test = None
            count = 0
            while x_test != None:
                x_test = x_test.to(self.device)
                x_fake_list = []
                x_fake_list.append(x_test)
                for j in range(self.c_dim):
                    for n in range(self.)
                    label = torch.ones((x_test.size(0),), dtype=torch.long).to(self.device)
                    label = label * j
                    x_gen = x_test.clone()
                    x_fake_list.append(self.gen_fake(x_gen, label, k, self.G_ema, self.M_ema, -1))  # remove iters
                try:
                    x_test, _ = next(loader[k])
                except:
                    x_test = None
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(dir_path, '{}images.jpg'.format(count))
                count += 1
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(sample_path))