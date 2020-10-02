import os
import argparse
from solver import Solver
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None

    # Solver for training and testing StarGAN.
    solver = Solver(config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=2, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=1, help='weight for reconstruction loss')
    parser.add_argument('--lambda_reg', type=float, default=1, help='weight for regularization loss')
    
    # Training configuration.
    #parser.add_argument('--dataset', type=str, default='RaFD', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--gpus', type=str, default='0', help="gpu id's to be used for training")
    parser.add_argument('--upsample_type', type=str, default='nearest', help='mode of upsampling')

    # Style parameters
    parser.add_argument('--latent_dim', type=int, default=16, help='dimension of the latent vector used in mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='dimension of the style vector returned by mapping network')

    # Progressive Training
    parser.add_argument('--iters', type=str, default='5000,7500,12500', help='iters to be performed per image size')
    parser.add_argument('--batch_size', type=str, default='128,64,32', help='batch size per image size')
    parser.add_argument('--img_size', type=str, default='64,128,256', help='image sizes to be used')
    parser.add_argument('--fade_point', type=float, default=0.5, help='fade point used for alpha')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--img_dir', type=str, default='data/train')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)