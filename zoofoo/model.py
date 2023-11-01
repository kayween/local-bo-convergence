import torch

import gpytorch
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.memoize import cached

from .rbf_kernel_directional_grad import RBFKernelDirectionalGrad
from .matern_kernel_directional_grad import MaternKernelDirectionalGrad


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.grad_covar_module = RBFKernelDirectionalGrad()

    @cached
    @torch.no_grad()
    def kernel_grad_grad(self, x):
        self.grad_covar_module.lengthscale = self.covar_module.base_kernel.lengthscale
        outputscale = self.covar_module.outputscale

        v1 = torch.eye(x.size(-1), device=x.device, dtype=x.dtype).repeat(x.size(-2), 1)

        covar = outputscale * self.grad_covar_module(x, v1=v1, v2=v1)
        covar = covar.evaluate()

        return covar

    def kernel_grad_func(self, x1, x2):
        self.grad_covar_module.lengthscale = self.covar_module.base_kernel.lengthscale
        outputscale = self.covar_module.outputscale

        v1 = torch.eye(x1.size(-1), device=x1.device, dtype=x1.dtype).repeat(
            x1.size(-2), 1
        )
        v2 = torch.eye(x2.size(-1), device=x2.device, dtype=x2.dtype)[0].repeat(
            x2.size(-2), 1
        )

        covar = outputscale * self.grad_covar_module(x1, x2, v1=v1, v2=v2)
        covar = covar.evaluate()[:, ::2]

        return covar

    @cached
    @torch.no_grad()
    def kernel_grad_train(self, x):
        return self.kernel_grad_func(x, self.train_inputs[0])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict_grad(self, x):
        self.grad_covar_module.lengthscale = self.covar_module.base_kernel.lengthscale
        outputscale = self.covar_module.outputscale

        v1 = torch.eye(x.size(-1), device=x.device, dtype=x.dtype).repeat(x.size(-2), 1)

        if self.train_inputs is None:
            predictive_covar = outputscale * self.grad_covar_module(x, v1=v1, v2=v1)
            predictive_mean = torch.zeros(
                predictive_covar.size(-2), device=x.device, dtype=x.dtype
            )
            predictive_mean[::x.size(-1) + 1] += self.mean_module.constant.item()
        else:
            v2 = torch.eye(x.size(-1), device=x.device, dtype=x.dtype)[0].repeat(
                self.train_inputs[0].size(-2), 1
            )

            test_train_covar = outputscale * self.grad_covar_module(
                x, self.train_inputs[0], v1=v1, v2=v2
            )
            test_train_covar = test_train_covar.evaluate()[
                :, ::2
            ]  # Weird indexing to get rid of unnneeded v2 info
            test_test_covar = outputscale * self.grad_covar_module(x, x, v1=v1, v2=v1)

            train_train_covar = self.covar_module(self.train_inputs[0])
            train_train_covar = train_train_covar.add_jitter(
                self.likelihood.noise
            )  # converts K -> K + \sigma ^2 I
            predictive_mean = test_train_covar @ train_train_covar.inv_matmul(
                self.train_targets - self.mean_module.constant.item()
            )
            predictive_mean[::x.size(-1) + 1] += self.mean_module.constant.item()

            predictive_covar = test_test_covar - test_train_covar @ (
                train_train_covar.inv_matmul(test_train_covar.transpose(-2, -1))
            )

        return gpytorch.distributions.MultivariateNormal(
            predictive_mean, predictive_covar
        )

    @cached
    @torch.no_grad()
    def cached_cholesky(self):
        train_train_covar = self.covar_module(self.train_inputs[0])
        train_train_covar = train_train_covar.add_jitter(
            self.likelihood.noise
        ).evaluate()
        return psd_safe_cholesky(train_train_covar)

    def posterior_covar(self, x, z_batch, grad_only=False):
        if self.train_inputs is None:
            K_xx = self.kernel_grad_grad(x)
            K_xz = self.kernel_grad_func(x, z_batch)
            K_zz = self.covar_module(z_batch)
            K_zz = K_zz.add_jitter(self.likelihood.noise).evaluate()

            L = psd_safe_cholesky(K_zz)

            return K_xx - K_xz.mm(torch.cholesky_solve(K_xz.T, L))

        else:
            L_tt = self.cached_cholesky()

            K_tz = self.covar_module(self.train_inputs[0], z_batch).evaluate()
            L_zt = torch.triangular_solve(K_tz, L_tt, upper=False).solution.T

            K_zz = self.covar_module(z_batch, z_batch)
            K_zz = K_zz.add_jitter(self.likelihood.noise).evaluate()
            L_zz = psd_safe_cholesky(K_zz - L_zt.mm(L_zt.T))

            zero = L_tt.new_zeros((L_tt.size(0), L_zz.size(1)))
            L = torch.cat(
                (torch.cat((L_tt, zero), dim=1), torch.cat((L_zt, L_zz), dim=1)), dim=0
            )

            K_xx = self.kernel_grad_grad(x)

            K_xt = self.kernel_grad_train(x)
            K_xz = self.kernel_grad_func(x, z_batch)
            K_x_tz = torch.cat((K_xt, K_xz), dim=1)

            posterior_covar = K_xx - K_x_tz.mm(torch.cholesky_solve(K_x_tz.T, L))

            if grad_only:
                return posterior_covar[1:, 1:]
            else:
                return posterior_covar

    def entropy(self, x, z_batch, grad_only=False):
        posterior_covar = self.posterior_covar(x, z_batch, grad_only)
        L = psd_safe_cholesky(posterior_covar)

        return L.diag().log().sum() * 2

    def nuclear_norm(self, x, z_batch, grad_only=False):
        posterior_covar = self.posterior_covar(x, z_batch, grad_only)
        return posterior_covar.diag().sum()

    def fro_norm(self, x, z_batch, grad_only=False):
        posterior_covar = self.posterior_covar(x, z_batch, grad_only)
        return 0.5 * (posterior_covar ** 2).sum()

    def information_gain(self, x, z_batch, predictive_covar):
        L = psd_safe_cholesky(predictive_covar)
        init_entropy = L.diag().log().sum() * 2

        new_entropy = self.entropy(x, z_batch)

        return init_entropy - new_entropy

    def grad_information_gain(self, x, z_batch, predictive_covar):
        L = psd_safe_cholesky(predictive_covar[..., 1:, 1:])
        init_entropy = L.diag().log().sum() * 2

        posterior_covar = self.posterior_covar(x, z_batch)[..., 1:, 1:]
        posterior_L = psd_safe_cholesky(posterior_covar)
        new_entropy = posterior_L.diag().log().sum() * 2

        return init_entropy - new_entropy


class MaternCovarModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.grad_covar_module = MaternKernelDirectionalGrad()
