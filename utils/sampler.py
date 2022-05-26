import math
import numpy as np
import torch
import torch.nn.functional as F


# 1d time-dependent Burgers
def burgers_train(args, device):
    # colocation points
    xc = torch.empty((args.nc, 1), device=device).uniform_(-1., 1.).requires_grad_()
    tc = torch.empty((args.nc, 1), device=device).uniform_(0., 1.).requires_grad_()
    # initial points
    xi = torch.empty((args.ni, 1), device=device).uniform_(-1., 1.)
    if args.model == 'pinn':
        ti = torch.zeros((args.ni, 1), device=device)
    else:
        ti = torch.zeros((1, 1), device=device)
    ui = -torch.sin(math.pi*xi)
    # boundary points
    if args.model == 'pinn':
        xb = torch.ones((args.nb, 1), device=device)
        xb[:args.nb//2, :] = -1
    else:
        xb = torch.tensor([[-1.], [1.]], device=device)
    tb = torch.empty((args.nb, 1), device=device).uniform_(0., 1.)
    return xc, tc, xi, ti, ui, xb, tb


def burgers_test(args, device):
    x = torch.linspace(-1., 1., args.nc_test, device=device)
    t = torch.linspace(0., 1., args.nc_test, device=device)
    if args.model == 'pinn':
        x, t = torch.meshgrid([x, t], indexing='ij')
    x, t = x.reshape(-1, 1), t.reshape(-1, 1)
    return x, t


# def burgers_test(args, data_dir):
#     gt_data = scipy.io.loadmat('burgers_shock.mat')
#     t = torch.tensor(gt_data['t'].flatten(), dtype=torch.float32)
#     x = torch.tensor(gt_data['x'].flatten(), dtype=torch.float32)
#     if args.model == 'pinn':
#         x, t = torch.meshgrid([x, t], indexing='ij')
#     x, t = x.reshape(-1, 1), t.reshape(-1, 1)
#     u_gt = torch.tensor(np.real(gt_data['usol']), dtype=torch.float32)
#     return x, t, u_gt


# 2D time-independent HelmHoltz
def _helmholtz2d_exact_u(a1, a2, y, x):
    return torch.sin(a1*math.pi*y) * torch.sin(a2*math.pi*x)


def _helmholtz2d_source_term(a1, a2, y, x, lda=1.):
    u = _helmholtz2d_exact_u(a1, a2, y, x)
    uyy = -(a1*math.pi)**2 * u
    uxx = -(a2*math.pi)**2 * u
    return  uyy + uxx + lda*u


def helmholtz2d_train(args, device):
    # colocation points
    yc = torch.empty((args.nc, 1), device=device).uniform_(-1., 1.).requires_grad_()
    xc = torch.empty((args.nc, 1), device=device).uniform_(-1., 1.).requires_grad_()
    with torch.no_grad():
        if args.model == 'pinn':
            uc = _helmholtz2d_source_term(args.a1, args.a2, yc, xc)
        else:
            yc_mesh, xc_mesh = torch.meshgrid(yc.view(-1), xc.view(-1), indexing='ij')
            uc = _helmholtz2d_source_term(args.a1, args.a2, yc_mesh, xc_mesh)
    # boundary points
    north = torch.empty((args.nb,), device=device).uniform_(-1., 1.)
    west = torch.empty((args.nb,), device=device).uniform_(-1., 1.)
    south = torch.empty((args.nb,), device=device).uniform_(-1., 1.)
    east = torch.empty((args.nb,), device=device).uniform_(-1., 1.)
    if args.model == 'pinn':
        ybs = torch.cat([
            torch.full((args.nb,), 1., device=device), west,
            torch.full((args.nb,), -1., device=device), east
        ]).reshape(-1, 1)
        xbs = torch.cat([
            north, torch.full((args.nb,), -1., device=device),
            south, torch.full((args.nb,), 1., device=device)
        ]).reshape(-1, 1)
        ubs = _helmholtz2d_exact_u(args.a1, args.a2, ybs, xbs)
    else:
        ybs = [torch.tensor([1.], device=device), west, torch.tensor([-1.], device=device), east]
        xbs = [north, torch.tensor([-1.], device=device), south, torch.tensor([1.], device=device)]
        ubs = []
        for i in range(4):
            yb, xb = torch.meshgrid(ybs[i], xbs[i], indexing='ij')
            ubs += [_helmholtz2d_exact_u(args.a1, args.a2, yb, xb)]
            ybs[i] = ybs[i].reshape(-1, 1)
            xbs[i] = xbs[i].reshape(-1, 1)
    return yc, xc, uc, ybs, xbs, ubs


def helmholtz2d_test(args, device):
    y = torch.linspace(-1., 1., args.nc_test, device=device)
    x = torch.linspace(-1., 1., args.nc_test, device=device)
    y_mesh, x_mesh = torch.meshgrid([y, x], indexing='ij')
    if args.model == 'pinn':
        y_mesh = y_mesh.reshape(-1, 1)
        x_mesh = x_mesh.reshape(-1, 1)
        u_gt = _helmholtz2d_exact_u(args.a1, args.a2, y_mesh, x_mesh)
        return y_mesh, x_mesh, u_gt
    else:
        y = y.reshape(-1, 1)
        x = x.reshape(-1, 1)
        u_gt = _helmholtz2d_exact_u(args.a1, args.a2, y_mesh, x_mesh)
        return y, x, u_gt


# 3D time-independent HelmHoltz
def _helmholtz3d_exact_u(a1, a2, a3, z, y, x):
    return torch.sin(a1*math.pi*z) * torch.sin(a2*math.pi*y) * torch.sin(a3*math.pi*x)


def _helmholtz3d_source_term(a1, a2, a3, z, y, x, lda=1.):
    u = _helmholtz3d_exact_u(a1, a2, a3, z, y, x)
    uzz = -(a1*math.pi)**2 * u
    uyy = -(a2*math.pi)**2 * u
    uxx = -(a3*math.pi)**2 * u
    return uzz + uyy + uxx + lda*u


def helmholtz3d_train(args, device):
    # colocation points
    zc = torch.empty((args.nc, 1), device=device).uniform_(-1., 1.).requires_grad_()
    yc = torch.empty((args.nc, 1), device=device).uniform_(-1., 1.).requires_grad_()
    xc = torch.empty((args.nc, 1), device=device).uniform_(-1., 1.).requires_grad_()
    with torch.no_grad():
        if args.model == 'pinn':
            uc = _helmholtz3d_source_term(args.a1, args.a2, args.a3, zc, yc, xc)
        else:
            zc_mesh, yc_mesh, xc_mesh = torch.meshgrid(zc.view(-1), yc.view(-1), xc.view(-1), indexing='ij')
            uc = _helmholtz3d_source_term(args.a1, args.a2, args.a3, zc_mesh, yc_mesh, xc_mesh)
    # boundary points
    def get_boundary_points(val):
        if val == 0:
            bnd = torch.empty((args.nb, 1), device=device, dtype=torch.float32).uniform_(-1., 1.)
        else:
            size = (args.nb, 1) if args.model == 'pinn' else (1, 1)
            if val == -1:
                bnd = torch.ones(size, device=device) * -1
            else:
                bnd = torch.ones(size, device=device)
        return bnd
    temp = F.one_hot(torch.tensor([2, 2, 1, 1, 0, 0]), 3)
    temp[1::2] *= -1
    zb, yb, xb = [], [], []
    for zi, yi, xi in temp:
        zb += [get_boundary_points(zi)]
        yb += [get_boundary_points(yi)]
        xb += [get_boundary_points(xi)]
    if args.model == 'pinn':
        zb = torch.cat(zb)
        yb = torch.cat(yb)
        xb = torch.cat(xb)
    return zc, yc, xc, uc, zb, yb, xb


def helmholtz3d_test(args, device):
    z = torch.linspace(-1, 1, args.nc_test, device=device)
    y = torch.linspace(-1, 1, args.nc_test, device=device)
    x = torch.linspace(-1, 1, args.nc_test, device=device)
    z_mesh, y_mesh, x_mesh = torch.meshgrid([z, y, x], indexing='ij')
    if args.model == 'pinn':
        z_mesh = z_mesh.reshape(-1, 1)
        y_mesh = y_mesh.reshape(-1, 1)
        x_mesh = x_mesh.reshape(-1, 1)
        u_gt = _helmholtz3d_exact_u(args.a1, args.a2, args.a3, z_mesh, y_mesh, x_mesh)
        return z_mesh, y_mesh, x_mesh, u_gt
    else:
        z = z.reshape(-1, 1)
        y = y.reshape(-1, 1)
        x = x.reshape(-1, 1)
        u_gt = _helmholtz3d_exact_u(args.a1, args.a2, args.a3, z_mesh, y_mesh, x_mesh)
        return z, y, x, u_gt


# 2D time-dependent heat & diffusion
def heat_diffusion_train(args, device):
    # colocation point
    tc = torch.rand((args.nc, 1), device=device, requires_grad=True)
    yc = torch.rand((args.nc, 1), device=device, requires_grad=True) * 2 - 1
    xc = torch.rand((args.nc, 1), device=device, requires_grad=True) * 2 - 1
    # initial points
    with torch.no_grad():
        yc_mesh, xc_mesh = torch.meshgrid(yc.view(-1), xc.view(-1), indexing='ij')
        ui = 0.25 * torch.exp(-((yc_mesh - 0.3)**2 + (xc_mesh - 0.2)**2) / 0.1)
        ui += 0.4 * torch.exp(-((yc_mesh + 0.5)**2 + (xc_mesh + 0.1)**2) * 15)
        ui += 0.3 * torch.exp(-(yc_mesh**2 + (xc_mesh + 0.5)**2) * 20)
        if args.model == 'pinn':
            ui = ui.reshape(-1, 1)
    return tc, yc, xc, ui


def heat_diffusion_test(args, device):
    t = torch.linspace(0, 1, args.nc_test, device=device)
    y = torch.linspace(-1, 1, args.nc_test, device=device)
    x = torch.linspace(-1, 1, args.nc_test, device=device)
    if args.model == 'spinn':
        t, y, x = t.reshape(-1, 1), y.reshape(-1, 1), x.reshape(-1, 1)
    return t, y, x