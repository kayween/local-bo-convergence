import math
import torch

from gpytorch.kernels.matern_kernel import MaternKernel

from .rbf_kernel_directional_grad import RBFKernelDirectionalGrad


class MaternKernelDirectionalGrad(MaternKernel, RBFKernelDirectionalGrad):
    def __init__(self, **kwargs):
        super().__init__(nu=2.5, **kwargs)  # forces nu = 2.5

    def forward(self, x1, x2, diag=False, shuffle=True, **params):
        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]

        v1 = params["v1"]
        v2 = params["v2"]
        # number of directions per point
        n_dir1 = int(v1.shape[-2] / n1)
        n_dir2 = int(v2.shape[-2] / n2)

        # set num the number of directions for num_outputs_per_input
        self.set_num_directions(n_dir1, n_dir2)

        # normalize directions
        v1 = (v1.T / torch.norm(v1, dim=1)).T
        v2 = (v2.T / torch.norm(v2, dim=1)).T

        K = torch.zeros(
            *batch_shape,
            n1 * (n_dir1 + 1),
            n2 * (n_dir2 + 1),
            device=x1.device,
            dtype=x1.dtype
        )

        if not diag:
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)

            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)
            constant_component = (
                (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            )
            constant_subcomponent = (math.sqrt(5) * distance).add(1)

            # 1) Kernel block
            K[..., :n1, :n2] = constant_component * exp_component

            # 2) First gradient block
            x2_v2 = x2_.reshape(n2, 1, d).bmm(
                torch.transpose(v2.reshape(n2, n_dir2, d), -2, -1)
            )
            x1_v2 = x1_ @ v2.T
            outer = x1_v2 - x2_v2.flatten()
            # permute cols so we get blocks for v1,v2,v3,...
            pi1 = (
                torch.arange(n2 * (n_dir2))
                .view(n2, n_dir2)
                .t()
                .reshape((n2 * (n_dir2)))
            )
            outer1 = outer[:, pi1] / self.lengthscale.unsqueeze(-2)

            K[..., :n1, n2:] = (
                5.0
                / 3.0
                * outer1
                * (constant_subcomponent * exp_component).repeat(
                    [*([1] * (n_batch_dims + 1)), n_dir2]
                )
            )

            # 3) Second gradient block
            x1_v1 = x1_.reshape(n1, 1, d).bmm(
                torch.transpose(v1.reshape(n1, n_dir1, d), -2, -1)
            )
            x2_v1 = x2_ @ v1.T
            outer = x1_v1.flatten() - x2_v1
            # permute cols so we get blocks for v1,v2,v3,...
            pi2 = (
                torch.arange(n1 * (n_dir1))
                .view(n1, n_dir1)
                .t()
                .reshape((n1 * (n_dir1)))
            )
            outer2 = outer[:, pi2]
            outer2 = outer2.t() / self.lengthscale.unsqueeze(-2)

            K[..., n1:, :n2] = (
                -5.0
                / 3.0
                * outer2
                * (constant_subcomponent * exp_component).repeat(
                    [n_dir1, *([1] * (n_batch_dims + 1))]
                )
            )

            # 4) Hessian block (n1*n_dir1, n2*n_dir2)
            outer3 = 5.0 * outer1.repeat(1, n_dir1, 1) * outer2.repeat(1, 1, n_dir2)

            # kronecker product term
            kp = v1 @ v2.T / self.lengthscale.pow(2)
            kp = kp[:, pi1][pi2, :]

            chain_rule = (
                kp * constant_component.repeat([*([1] * n_batch_dims), n_dir1, n_dir2])
                - outer3
            )
            K[..., n1:, n2:] = (
                5.0
                / 3.0
                * chain_rule
                * exp_component.repeat([*([1] * n_batch_dims), n_dir1, n_dir2])
            )

            # Apply a perfect shuffle permutation to match the MutiTask ordering
            if shuffle:
                pi1 = (
                    torch.arange(n1 * (n_dir1 + 1))
                    .view(n_dir1 + 1, n1)
                    .t()
                    .reshape((n1 * (n_dir1 + 1)))
                )
                pi2 = (
                    torch.arange(n2 * (n_dir2 + 1))
                    .view(n_dir2 + 1, n2)
                    .t()
                    .reshape((n2 * (n_dir2 + 1)))
                )
                K = K[..., pi1, :][..., :, pi2]
            return K

        else:
            if not (
                n1 == n2
                and torch.eq(x1, x2).all()
                and n_dir1 == n_dir2
                and torch.eq(v1, v2).all()
            ):
                raise RuntimeError("diag=True only works when x1 == x2 and v1 == v2")

            kernel_diag = super().forward(x1, x2, diag=True)

            grad_diag = (
                5.0
                / 3.0
                * torch.ones(*batch_shape, n2, n_dir2, device=x1.device, dtype=x1.dtype)
                / self.lengthscale.pow(2)
            )
            grad_diag = grad_diag.transpose(-1, -2).reshape(*batch_shape, n2 * n_dir2)

            k_diag = torch.cat((kernel_diag, grad_diag), dim=-1)
            if shuffle:
                pi = (
                    torch.arange(n2 * (n_dir2 + 1))
                    .view(n_dir2 + 1, n2)
                    .t()
                    .reshape((n2 * (n_dir2 + 1)))
                )

                return k_diag[..., pi]
            return k_diag
