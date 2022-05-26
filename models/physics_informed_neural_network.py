import torch
import torch.nn as nn
from models.base_network import BaseNetwork
from models.residual_function import *


class PINN2D(nn.Module):
    def __init__(self, features, activation, equation):
        super(PINN2D, self).__init__()
        self.fyx = BaseNetwork(features, 'linear', activation, True)
        self.resid_fn = setup_residual2d_reverse_mode(self.forward, equation)
    
    def forward(self, y, x):
        return self.fyx(torch.cat([y, x], dim=-1))
    

class SPINN2D(nn.Module):
    def __init__(self, features, activation, equation):
        super(SPINN2D, self).__init__()
        self.fy = BaseNetwork(features, 'linear', activation, True)
        self.fx = BaseNetwork(features, 'linear', activation, True)
        self.resid_fn = setup_residual2d_forward_mode(self.forward, equation)
    
    def forward(self, y, x):
        return torch.mm(self.fy(y), self.fx(x).T)


class PINN3D(nn.Module):
    def __init__(self, features, activation, equation, meshgrid=True):
        super(PINN3D, self).__init__()
        self.fzyx = BaseNetwork(features, 'linear', activation, True)
        self.meshgrid = meshgrid
        if meshgrid:
            self.resid_fn = setup_residual3d_reverse_mode(self.fzyx, equation)
        else:
            self.resid_fn = setup_residual3d_reverse_mode(self.forward, equation)

    def forward(self, z, y, x):
        if self.meshgrid:
            z, y, x = torch.meshgrid([z.view(-1), y.view(-1), x.view(-1)], indexing='ij')
            z, y, x = z.reshape(-1, 1), y.reshape(-1, 1), x.reshape(-1, 1)
        return self.fzyx(torch.cat([z, y, x], dim=-1))
    

class SPINN3D(nn.Module):
    def __init__(self, features, activation, equation):
        super(SPINN3D, self).__init__()
        self.fx = BaseNetwork(features, 'linear', activation, True)
        self.fy = BaseNetwork(features, 'linear', activation, True)
        self.fz = BaseNetwork(features, 'linear', activation, True)
        self.resid_fn = setup_residual3d_forward_mode(self.forward, equation)
    
    def forward(self, z, y, x):
        z = self.fz(z).permute(1, 0)
        y = self.fy(y).permute(1, 0)
        x = self.fx(x).permute(1, 0)
        zy = torch.einsum('fz, fy->fzy', z, y)
        return torch.einsum('fzy, fx->zyx', zy, x)