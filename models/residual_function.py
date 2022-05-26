import math
import torch
from torch.autograd import grad
from torch.autograd.functional import jvp
from models.base_network import hvp_fwd


def setup_residual2d_reverse_mode(apply_fn, equation):
    def burgers(x, t, nu=0.01/math.pi):
        u = apply_fn(x, t)
        ut = grad(u, t, torch.ones_like(u), True, True)[0]
        ux = grad(u, x, torch.ones_like(u), True, True)[0]
        uxx = grad(ux, x, torch.ones_like(ux), True, True)[0]
        return ut + u*ux - nu*uxx
    def helmholtz(y, x, lda=1.):
        u = apply_fn(y, x)
        uy = grad(u, y, torch.ones_like(u), True, True)[0]
        uyy = grad(uy, y, torch.ones_like(uy), True, True)[0]
        ux = grad(u, x, torch.ones_like(u), True, True)[0]
        uxx = grad(ux, x, torch.ones_like(ux), True, True)[0]
        return uyy + uxx + lda*u
    if equation == 'burgers':
        return burgers
    elif equation == 'helmholtz2d':
        return helmholtz
    else:
        raise ValueError('Invalid equation.')


def setup_residual2d_forward_mode(apply_fn, equation):
    def burgers(x, t, nu=0.01/math.pi):
        # make func dependent of only 1 var
        ft = lambda t: apply_fn(x, t)
        fx = lambda x: apply_fn(x, t)
        # forward mode AD
        u = apply_fn(x, t)
        ut = jvp(ft, t, torch.ones_like(t), True)[1]
        ux, uxx = hvp_fwd(fx, x, torch.ones_like(x), torch.ones_like(x), True)
        return ut + u*ux - nu*uxx
    def helmholtz(y, x, lda=1.):
        # make func dependent on only 1 var
        fy = lambda y: apply_fn(y, x)
        fx = lambda x: apply_fn(y, x)
        # forward mode AD
        u = apply_fn(y, x)
        uyy = hvp_fwd(fy, y, torch.ones_like(y), torch.ones_like(y))
        uxx = hvp_fwd(fx, x, torch.ones_like(x), torch.ones_like(x))
        return uyy + uxx + lda*u
    if equation == 'burgers':
        return burgers
    elif equation == 'helmholtz2d':
        return helmholtz
    else:
        raise ValueError('Invalid equation.')


def setup_residual3d_reverse_mode(apply_fn, equation):
    def heat_diffusion(t, y, x):
        t, y, x = torch.meshgrid([t.view(-1), y.view(-1), x.view(-1)], indexing='ij')
        t, y, x = t.reshape(-1, 1), y.reshape(-1, 1), x.reshape(-1, 1)
        u = apply_fn(torch.cat([t, y, x], dim=-1))
        ut = grad(u, t, torch.ones_like(u), True, True)[0]
        uy = grad(u, y, torch.ones_like(u), True, True)[0]
        uyy = grad(uy, y, torch.ones_like(uy), True, True)[0]
        ux = grad(u, x, torch.ones_like(u), True, True)[0]
        uxx = grad(ux, x, torch.ones_like(ux), True, True)[0]
        if equation == 'heat':
            return ut - 0.05 * (uxx + uyy)
        else:
            return ut - 0.05 * (ux**2 + u*uxx + uy**2 + u*uyy)
    def helmholtz(z, y, x, lda=1.):
        u = apply_fn(z, y, x)
        uz = torch.autograd.grad(u, z, torch.ones_like(u), True, True)[0]
        uzz = torch.autograd.grad(uz, z, torch.ones_like(uz), True, True)[0]
        uy = torch.autograd.grad(u, y, torch.ones_like(u), True, True)[0]
        uyy = torch.autograd.grad(uy, y, torch.ones_like(uy), True, True)[0]
        ux = torch.autograd.grad(u, x, torch.ones_like(u), True, True)[0]
        uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), True, True)[0]
        return uzz + uyy + uxx + lda*u
    if equation == 'heat' or equation == 'diffusion':
        return heat_diffusion
    elif equation == 'helmholtz3d':
        return helmholtz
    else:
        raise NotImplementedError


def setup_residual3d_forward_mode(apply_fn, equation):
    def heat_diffusion(t, y, x):
        # make func dependent of only 1 var
        ft = lambda t: apply_fn(t, y, x)
        fy = lambda y: apply_fn(t, y, x)
        fx = lambda x: apply_fn(t, y, x)
        # forward mode AD
        ut = jvp(ft, t, torch.ones_like(t), True)[1]
        if equation == 'heat':
            uyy = hvp_fwd(fy, y, torch.ones_like(y), torch.ones_like(y))
            uxx = hvp_fwd(fx, x, torch.ones_like(x), torch.ones_like(x))
            return ut - 0.05 * (uxx + uyy)
        else:
            u = apply_fn(t, y, x)
            uy, uyy = hvp_fwd(fy, y, torch.ones_like(y), torch.ones_like(y), True)
            ux, uxx = hvp_fwd(fx, x, torch.ones_like(x), torch.ones_like(x), True)
            return ut - 0.05 * (ux**2 + u*uxx + uy**2 + u*uyy)
    def helmholtz(z, y, x, lda=1.):
        # make func dependent on only 1 var
        fz = lambda z: apply_fn(z, y, x)
        fy = lambda y: apply_fn(z, y, x)
        fx = lambda x: apply_fn(z, y, x)
        # calc loss (forward mode)
        u = apply_fn(z, y, x)
        uzz = hvp_fwd(fz, z, torch.ones_like(z), torch.ones_like(z))
        uyy = hvp_fwd(fy, y, torch.ones_like(y), torch.ones_like(y))
        uxx = hvp_fwd(fx, x, torch.ones_like(x), torch.ones_like(x))
        return uzz + uyy + uxx + lda*u
    if equation == 'heat' or equation == 'diffusion':
        return heat_diffusion
    elif equation == 'helmholtz3d':
        return helmholtz
    else:
        raise NotImplementedError

