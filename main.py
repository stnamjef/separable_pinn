import os
import argparse
import math
import torch
from trainer import *
from utils.util import *
from models.physics_informed_neural_network import *


if __name__ =='__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')
    
    # pde settings
    parser.add_argument('--eq', type=str, default='helmholtz2d', help='PDE to solve')
    parser.add_argument('--nc', type=int, default=10000, help='the number of colocation coordinate')
    parser.add_argument('--ni', type=int, default=50, help='the number of initial condition coordinate')
    parser.add_argument('--nb', type=int, default=50, help='the number of boundary condition coordinate')
    parser.add_argument('--nc_test', type=int, default=100, help='the number of colocation coordinate in test time')
    # parser.add_argument('--boundary', type=str, default='0,1,-1,1', help='lower and upper bounds')

    # burgers
    parser.add_argument('--nu', type=int, default=0.01/math.pi, help='viscosity in Burgers equation')
    parser.add_argument('--lda', type=int, default=1, help='lambda*initial_loss + boundary_loss + pde_loss')
    
    # helmholtz
    parser.add_argument('--a1', type=int, default=4, help='sin(a1*pi*y)+sin(a2*pi*x)')
    parser.add_argument('--a2', type=int, default=1, help='sin(a1*pi*y)+sin(a2*pi*x)')
    parser.add_argument('--a3', type=int, default=2, help='sin(a1*pi*z)+sin(a2*pi*y)+sin(a3*pi*x)')

    # training settings
    parser.add_argument('--seed', type=int, default=444, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')
    parser.add_argument('--model', type=str, default='spinn', help='pinn, spinn')
    parser.add_argument('--n_layers', type=int, default=9, help='the number of layer')
    parser.add_argument('--in_size', type=int, default=20, help='input feature size')
    parser.add_argument('--out_size', type=int, default=20, help='output feature size')
    parser.add_argument('--hidden_size', type=int, default=20, help='hidden feature size')

    # log settings
    parser.add_argument('--root_dir', type=str, default=None, help='root directory')
    parser.add_argument('--log_iter', type=int, default=100, help='print log every...')
    parser.add_argument('--plot_iter', type=int, default=10000, help='plot result every...')

    args = parser.parse_args()

    # remove randomness
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # make model & optimizer
    features = concat_features(args.in_size, args.hidden_size, args.out_size, args.n_layers)

    if args.eq in ['burgers', 'helmholtz2d']:
        if args.model == 'pinn':
            model = PINN2D(features, 'tanh', args.eq).to(device)
        elif args.model == 'spinn':
            model = SPINN2D(features, 'tanh', args.eq).to(device)
        else:
            raise ValueError('Invalid model.')
    else:
        if args.model == 'pinn':
            model = PINN3D(features, 'tanh', args.eq, args.eq != 'helmholtz3d').to(device)
        elif args.model == 'spinn':
            model = SPINN3D(features, 'tanh', args.eq).to(device)
        else:
            raise ValueError('Invalid model.')
    
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # concat experiment config
    config = concat_config(args, count_total_params(model))

    # set directory
    if args.root_dir == None:
        args.root_dir = os.getcwd()
    result_dir = os.path.join(args.root_dir, 'results', args.eq, args.model, config)

    # make dir
    os.makedirs(result_dir, exist_ok=True)

    if args.eq == 'burgers':
        train_burgers(args, device, model, optim, result_dir)
    elif args.eq == 'helmholtz2d':
        train_helmholtz2d(args, device, model, optim, result_dir)
    elif args.eq == 'helmholtz3d':
        train_helmholtz3d(args, device, model, optim, result_dir)
    elif args.eq == 'heat':
        train_heat_diffusion(args, device, model, optim, result_dir)
    elif args.eq == 'diffusion':
        train_heat_diffusion(args, device, model, optim, result_dir)
    else:
        raise ValueError('Invalid equation')
