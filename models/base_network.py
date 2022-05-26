import math
import torch
import torch.nn as nn
from torch.autograd.functional import jvp


def _setup_layers(features, ltype, kernel_size, stride, padding):
    layers = []
    for m, n in zip(features[:-1], features[1:]):
        if ltype == 'linear':
            layers += [nn.Linear(m, n)]
        elif ltype == 'conv2d':
            layers += [nn.Conv2d(m, n, kernel_size, stride, padding)]
        else:
            raise ValueError('Invalid layer type.')
    return nn.ModuleList(layers)


def _setup_activation(activation):
    if activation == 'tanh':
        func = nn.Tanh()
    elif activation == 'relu':
        func = nn.ReLU()
    elif activation == 'gelu':
        func = nn.GELU()
    elif activation == 'sine':
        func = torch.sin
    else:
        raise ValueError('Invalid activation.')
    return func


def _calculate_fan_in(w):
    n_input_fmaps = w.size(1)
    receptive_field_size = 1
    if w.dim() > 2:
        for s in w.shape[2:]:
            receptive_field_size *= s
    fan_in = n_input_fmaps * receptive_field_size
    return fan_in


def _init_siren(w, w0, is_first):
        fan_in = _calculate_fan_in(w)
        u = 1/fan_in if is_first else math.sqrt(6/fan_in)/w0
        nn.init.uniform_(w, -u, u)


def _initialize_layers(layers, itype, w0=30.):
    for i, layer in enumerate(layers):
        if itype == 'tanh':
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif itype == 'relu':
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        elif itype == 'gelu':
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        elif itype == 'sine':
            _init_siren(layer.weight, w0, i==0)
        else:
            raise ValueError('Invalid initialization.')


class BaseNetwork(nn.Module):
    def __init__(self, features, ltype, activation, outermost_linear,
                kernel_size=3, stride=1, padding=1):
        super(BaseNetwork, self).__init__()
        self.layers = _setup_layers(features, ltype, kernel_size, stride, padding)
        self.activate = _setup_activation(activation)
        self.outermost_linear = outermost_linear
        _initialize_layers(self.layers, activation)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activate(layer(x))
        x = self.layers[-1](x)
        return x if self.outermost_linear else self.activate(x)


def hvp_fwd(f, primals, f_tangents, g_tangents, aux=False):
    g = lambda primals: jvp(f, primals, f_tangents, True)
    x_dx, xx_dxx = jvp(g, primals, g_tangents, True)
    if aux == False:
        return xx_dxx[1]
    else:
        return x_dx[1], xx_dxx[1]