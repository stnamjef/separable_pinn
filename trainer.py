import os
import torch
from tqdm import trange
from utils.plot import *
from utils.sampler import *
from utils.util import squared_mean, relative_l2


def train_burgers(args, device, model, optim, result_dir):
    x, t = burgers_test(args, device)
    for e in trange(1, args.epochs+1):
        xc, tc, xi, ti, ui, xb, tb = burgers_train(args, device)
        l1 = squared_mean((model(xi, ti) - ui))
        l2 = squared_mean(model(xb, tb))
        l3 = squared_mean(model.resid_fn(xc, tc))
        loss = args.lda*l1 + l2 + l3
        optim.zero_grad()
        loss.backward()
        optim.step()
        if e % args.log_iter == 0:
            log = f'Epoch: {e}/{args.epochs} --> initial: {l1.item():.8f}, ' + \
                  f'boundary: {l2.item():.8f}, domain: {l3.item():.8f}, total: {loss.item():.8f}'
            print(log)
            with open(os.path.join(result_dir, 'log.csv'), 'a') as f:
                f.write(f'{l1.item():.8f}, {l2.item():.8f}, {l3.item():.8f}, {loss.item():.8f}\n')
        if e % args.plot_iter == 0:
            with torch.no_grad():
                plot_burgers(t, x, model(x, t), e, result_dir, args.nc_test)
    print('Training done.')
    

def train_helmholtz2d(args, device, model, optim, result_dir):
    yc, xc, uc, yb, xb, ub = helmholtz2d_train(args, device)
    y, x, u_gt = helmholtz2d_test(args, device)
    for e in trange(1, args.epochs+1):
        if args.model == 'pinn':
            l1 = squared_mean(model(yb, xb) - ub)
        else:
            l1 = 0.
            for i in range(4):
                l1 += 0.25 * squared_mean(model(yb[i], xb[i]) - ub[i])
        l2 = squared_mean(model.resid_fn(yc, xc) - uc)
        loss = l1 + l2
        optim.zero_grad()
        loss.backward()
        optim.step()
        if e % args.log_iter == 0:
            log = f'Epoch: {e}/{args.epochs} --> boundary: {l1.item():.8f}, ' + \
                  f'domain: {l2.item():.8f}, total: {loss.item():.8f}'
            print(log)
            with open(os.path.join(result_dir, 'log.csv'), 'a') as f:
                f.write(f'{l1.item():.8f}, {l2.item():.8f}, {loss.item():.8f}\n')
        if e % args.plot_iter == 0:
            with torch.no_grad():
                plot_helmholtz2d(y, x, model(y, x), u_gt, e, result_dir, args.nc_test)
    print('Training done.')
    with torch.no_grad():
        error = relative_l2(model(y, x), u_gt)
        print(f'Relative L2: {error:.4f}')


def train_helmholtz3d(args, device, model, optim, result_dir):
    zc, yc, xc, uc, zb, yb, xb = helmholtz3d_train(args, device)
    z, y, x, u_gt = helmholtz3d_test(args, device)

    for e in trange(1, args.epochs+1):
        if args.model == 'pinn':
            l1 = squared_mean(model(zb, yb, xb))
        else:
            l1 = 0.
            for i in range(6):
                l1 += (1/6) * squared_mean(model(zb[i], yb[i], xb[i]))
        l2 = squared_mean(model.resid_fn(zc, yc, xc) - uc)
        loss = l1 + l2
        optim.zero_grad()
        loss.backward()
        optim.step()
        if e % args.log_iter == 0:
            log = f'Epoch: {e}/{args.epochs} --> boundary: {l1.item():.8f}, ' + \
                  f'domain: {l2.item():.8f}, total: {loss.item():.8f}'
            print(log)
            with open(os.path.join(result_dir, 'log.csv'), 'a') as f:
                f.write(f'{l1.item():.8f}, {l2.item():.8f}, {loss.item():.8f}\n')
        if e % args.plot_iter == 0:
            with torch.no_grad():
                plot_helmholtz3d(z, y, x, model(z, y, x), e, result_dir)
    print('Training done.')
    with torch.no_grad():
        error = relative_l2(model(z, y, x), u_gt)
        print(f'Relative L2: {error:.4f}')


def train_heat_diffusion(args, device, model, optim, result_dir):
    t, y, x = heat_diffusion_test(args, device)
    for e in trange(1, args.epochs+1):
        tc, yc, xc, ui = heat_diffusion_train(args, device)
        # initial loss
        l1 = squared_mean(model(torch.tensor([[0.]], device=device), yc, xc) - ui)
        # boundary loss
        l2 = 0
        l2 += squared_mean(model(tc, yc, torch.tensor([[-1.]], device=device)))
        l2 += squared_mean(model(tc, yc, torch.tensor([[1.]], device=device)))
        l2 += squared_mean(model(tc, torch.tensor([[-1.]], device=device), xc))
        l2 += squared_mean(model(tc, torch.tensor([[1.]], device=device), xc))
        # domain loss
        l3 = squared_mean(model.resid_fn(tc, yc, xc))
        # total loss
        loss = l1 + l2 + l3
        optim.zero_grad()
        loss.backward()
        optim.step()
        if e % args.log_iter == 0:
            # print & write log
            log = f'Epoch: {e}/{args.epochs} --> initial: {l1.item():.8f},' + \
                  f'boundary: {l2.item():.8f}, domain: {l3.item():.8f}, total: {loss.item():.8f}'
            print(log)
            with open(os.path.join(result_dir, 'log.csv'), 'a') as f:
                f.write(f'{l1.item():.8f}, {l2.item():.8f}, {l3.item():.8f}, {loss.item():.8f}\n')
        if e % args.plot_iter == 0:
            with torch.no_grad():
                # plot directory
                plot_dir = os.path.join(result_dir, 'plot', f'{e}')
                os.makedirs(plot_dir, exist_ok=True)
                # eval and plot
                u = model(t, y, x).reshape(args.nc_test, args.nc_test, args.nc_test)
                plot_heat_diffusion(y, x, u, plot_dir)
    print('Training done.')
    tt, error = 0., 0.
    data_dir = os.path.join(args.root_dir, 'data', args.eq)
    with torch.no_grad():
        u = model(t, y, x).reshape(args.nc_test, args.nc_test, args.nc_test)
        for i in range(args.nc_test):
            u_gt = np.load(os.path.join(data_dir, f'heat_gaussian_{tt:.2f}.npy'))
            u_gt = torch.tensor(u_gt).cuda()
            error += relative_l2(u[i], u_gt)
            tt += 0.01
    error /= args.nc_test
    print(f'Relative L2: {error:.4f}')
