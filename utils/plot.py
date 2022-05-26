import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_burgers(t, x, u, e, plot_dir, nc):
    if t.numel() != u.numel():
        x, t = torch.meshgrid([x.view(-1), t.view(-1)], indexing='ij')
    # ship back to cpu
    t = t.cpu().numpy().reshape(nc, nc)
    x = x.cpu().numpy().reshape(nc, nc)
    u = u.detach().cpu().numpy().reshape(nc, nc)
    # plotting
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, u, cmap='rainbow', shading='auto')
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t, x)')
    cbar.mappable.set_clim(-1, 1)
    for i, t_cs in enumerate([0, 0.5, 1.0]):
        idx = 0 if t_cs == 0 else int(t_cs*nc)-1
        plt.subplot(gs[1, i])
        plt.plot(u[:, idx])
        plt.title(f't={t_cs}')
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    # save fig
    fig.savefig(os.path.join(plot_dir, f'solution_{e}.jpg'))
    # clear fig
    plt.clf()


def plot_helmholtz2d(y, x, u, u_gt, e, plot_dir, nc):
    if y.numel() != u.numel():
        y, x = torch.meshgrid([y.view(-1), x.view(-1)], indexing='ij')
    # ship back to cpu
    y = y.cpu().numpy().reshape(nc, nc)
    x = x.cpu().numpy().reshape(nc, nc)
    u = u.cpu().numpy().reshape(nc, nc)
    u_gt = u_gt.cpu().numpy().reshape(nc, nc)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].set_aspect('equal')
    col0 = axes[0].pcolormesh(x, y, u_gt, cmap='rainbow', shading='auto')
    axes[0].set_xlabel('x', fontsize=12, labelpad=12)
    axes[0].set_ylabel('y', fontsize=12, labelpad=12)
    axes[0].set_title('Exact U', fontsize=18, pad=18)
    div0 = make_axes_locatable(axes[0])
    cax0 = div0.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col0, cax=cax0)

    axes[1].set_aspect('equal')
    col1 = axes[1].pcolormesh(x, y, u, cmap='rainbow', shading='auto')
    axes[1].set_xlabel('x', fontsize=12, labelpad=12)
    axes[1].set_ylabel('y', fontsize=12, labelpad=12)
    axes[1].set_title('Predicted U', fontsize=18, pad=18)
    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col1, cax=cax1)

    axes[2].set_aspect('equal')
    col2 = axes[2].pcolormesh(x, y, np.abs(u-u_gt), cmap='rainbow', shading='auto')
    axes[2].set_xlabel('x', fontsize=12, labelpad=12)
    axes[2].set_ylabel('y', fontsize=12, labelpad=12)
    axes[2].set_title('Absolute error', fontsize=18, pad=18)
    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col2, cax=cax2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'solution_{e}.jpg'))
    plt.clf()


def plot_helmholtz3d(z, y, x, u, e, plot_dir):
    if z.numel() != u.numel():
        z, y, x = torch.meshgrid(z.view(-1), y.view(-1), x.view(-1), indexing='ij')
    z = z.cpu().numpy().flatten()
    y = y.cpu().numpy().flatten()
    x = x.cpu().numpy().flatten()
    u = u.cpu().numpy().flatten()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z, y, x, c=u, s=0.5, cmap='Spectral')
    plt.savefig(os.path.join(plot_dir, f'solution_{e}.png'))


def plot_heat_diffusion(y, x, u, plot_dir):
    y_mesh, x_mesh = torch.meshgrid(y.view(-1), x.view(-1), indexing='ij')
    y_mesh = y_mesh.cpu().numpy()
    x_mesh = x_mesh.cpu().numpy()
    for i in [0, 25, 50, 100]:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.view_init(azim=280)
        ax.set_zlim(0, 0.41)
        sf = ax.plot_surface(
            x_mesh, y_mesh, u[i].cpu().numpy(), vmin=0, vmax=0.41,
            cmap='coolwarm', linewidth=0, antialiased=False
        )
        fig.colorbar(sf, shrink=0.5)
        plt.savefig(os.path.join(plot_dir, f'{i}.jpg'))
        plt.close()
