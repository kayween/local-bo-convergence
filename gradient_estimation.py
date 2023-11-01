import random

import numpy as np

import torch

from utils import create_model
from utils import update_model
# from utils import update_train_data
from utils import get_query_points

import matplotlib.pyplot as plt
import seaborn as sns


def f(x):
    center = torch.tensor([1., 1.], dtype=x.dtype, device=x.device).unsqueeze(-2)
    return 0.5 * (x - center).square().sum(dim=-1)


def relu(x, noise_std=1e-2):
    n, d = x.size()
    return torch.relu(x).squeeze(-1) + noise_std * torch.randn(n, device=x.device)


def test(objective, x, batch_size, device):
    model, likelihood = create_model(None, None)
    model.to(device), likelihood.to(device)

    model.covar_module.outputscale = 1.
    model.covar_module.base_kernel.lengthscale = 1.
    likelihood.noise = 2e-4

    train_x, _ = get_query_points(model, likelihood, x, batch_size=batch_size)
    train_y = objective(train_x)
    print(train_x.size())
    print(train_y.size())

    model, likelihood = create_model(train_x, train_y)
    model.train().to(device), likelihood.train().to(device)

    update_model(model, likelihood, train_x, train_y)

    model.eval(), likelihood.eval()
    with torch.no_grad():
        g = model.predict_grad(x)

    # very close to [-1., -1.]
    print("nabla mu", g.mean[1:])
    print(g.covariance_matrix)
    print("trace", g.covariance_matrix[1:, 1:].diag().sum().item())
    return g, train_x, train_y


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    seed = 4321

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda:0"

    x = torch.tensor([0.], device=device).unsqueeze(-2)

    g, train_x, train_y = test(objective=relu, x=x, batch_size=4, device=device)

    xx = np.linspace(-2, 1, num=1000)
    yy = relu(torch.tensor(xx).unsqueeze(-1), noise_std=0.)

    sns.set_theme()
    sns.set_context("paper")
    sns.set_style("darkgrid")

    plt.plot(xx, yy, label='relu')
    plt.scatter(train_x.squeeze().cpu().numpy(), train_y.cpu().numpy(), c='r', label='queries')
    plt.legend(fontsize=12)

    plt.show()
