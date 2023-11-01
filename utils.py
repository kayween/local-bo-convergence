import math

from tqdm import tqdm

import torch

import gpytorch
from zoofoo.model import ExactGPModel, MaternCovarModel

from zoofoo.BFGS.LBFGS import FullBatchLBFGS


def create_model(train_x, train_y, kernel='matern'):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if kernel == 'gaussian':
        model = ExactGPModel(train_x, train_y, likelihood)
    elif kernel == 'matern':
        model = MaternCovarModel(train_x, train_y, likelihood)

    return model, likelihood


def get_query_points(model, likelihood, x, batch_size):
    """
    Args:
        x (tensor): 1 x d tensor
    """
    model.eval().to(x.device)
    likelihood.eval().to(x.device)

    d = x.size(-1)

    z = x + 1e-4 / math.sqrt(d) * torch.randn(batch_size, d, dtype=x.dtype, device=x.device)
    z.requires_grad_(True)

    optimizer = FullBatchLBFGS([z], lr=1e-1)

    def closure():
        optimizer.zero_grad()
        loss = model.nuclear_norm(x, z, grad_only=True)
        return loss

    loss = closure()
    loss.backward()

    lst_loss = []
    cnt_small_updates = 0

    train_iter = tqdm(range(200))
    for i in train_iter:
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 20}
        loss, _, lr, ls_step, F_eval, G_eval, _, fail = optimizer.step(options)

        train_iter.set_postfix(
            loss=loss.item(),
            lr=lr,
            ls_step=ls_step,
        )
        lst_loss.append(loss.item())

        if fail:
            print("Warning: line search fails. Taking step size {:f}".format(lr))

        if i > 5:
            if lst_loss[-5] - lst_loss[-1] < 1e-5:
                cnt_small_updates += 1
            else:
                cnt_small_updates = 0

            if cnt_small_updates >= 3:
                break

    return z.clone().detach(), loss.item()


def update_model(model, likelihood, train_x, train_y):
    model.train()
    likelihood.train()

    optimizer = FullBatchLBFGS(model.parameters(), lr=1.)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def closure():
        optimizer.zero_grad()
        with gpytorch.settings.fast_computations(False, False, False):
            output = model(train_x)
            loss = -mll(output, train_y)
        return loss

    loss = closure()
    loss.backward()

    train_iter = tqdm(range(50))
    for i in train_iter:
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 20}
        loss, _, lr, _, F_eval, G_eval, _, fail = optimizer.step(options)

        train_iter.set_postfix(
            loss=loss.item(),
            ls=model.covar_module.base_kernel.lengthscale.mean().item(),
            os=model.covar_module.outputscale.item(),
            sn=model.likelihood.noise.item()
        )
