import random

import numpy as np

import torch

import matplotlib.pyplot as plt
import seaborn as sns

from gradient_estimation import test


def ff(x, noise_std=1e-2):
    n, d = x.size()
    center = torch.tensor([0., 0.], dtype=x.dtype, device=x.device).unsqueeze(-2)
    return 0.5 * (x - center).square().sum(dim=-1) + noise_std * torch.randn(n, device=x.device)


def l1(x, noise_std=1e-2):
    n, d = x.size()
    return x.abs().sum(dim=-1) + noise_std * torch.randn(n, device=x.device)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    seed = 4321

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda:0"

    x = torch.tensor([0., 1.], device=device).unsqueeze(-2)

    g, train_x, train_y = test(objective=lambda x: ff(x, noise_std=0.01), x=x, batch_size=5, device=device)
    # g, train_x, train_y = test(objective=lambda x: ff(x, noise_std=0.01), x=x, batch_size=10, device=device)
    # g, train_x, train_y = test(objective=lambda x: l1(x, noise_std=0.01), x=x, batch_size=5, device=device)
    # g, train_x, train_y = test(objective=lambda x: l1(x, noise_std=0.01), x=x, batch_size=10, device=device)

    plt.figure(figsize=(6, 5))

    sns.set_theme()
    sns.set_context("paper")
    sns.set_style("darkgrid")

    g = g.mean[1:].cpu().numpy()
    x = x.cpu().numpy()
    train_x = train_x.cpu().numpy()

    # plt.plot(np.linspace(-1, 1), [1] * 50, color='b', linestyle='--', linewidth=3, label='subgradient')
    plt.scatter(train_x[:, 0], train_x[:, 1], s=30, c='g', label=r'data $\mathcal{D}$')
    plt.scatter(x[0, 0], x[0, 1], marker='*', s=150, c='r', label=r'$\mathbf{x}$')

    xx = np.linspace(-1.5, 1.5, num=500)
    yy = np.linspace(-1, 2, num=500)
    x, y = np.meshgrid(xx, yy)
    levels = 0.5 * (x ** 2 + y ** 2)
    # levels = abs(x) + abs(y)
    plt.contour(x, y, levels, 6)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1, 2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=20)

    plt.tight_layout()
    plt.show()
