import math
import random

import numpy as np

import torch

from utils import create_model
from utils import get_query_points

from zoofoo.BFGS.LBFGS import FullBatchLBFGS

from tqdm import tqdm


def evaluate_error_functions(batch, d, kernel, noise, device):
    model, likelihood = create_model(None, None)
    model.to(device), likelihood.to(device)

    model.covar_module.outputscale = 1.
    model.covar_module.base_kernel.lengthscale = 1.
    likelihood.noise = noise

    x = torch.zeros(1, d, device=device)

    min_value = math.inf
    for i in range(5):
        _, value = get_query_points(model, likelihood, x, batch_size=batch)
        min_value = min(min_value, value)

        print("batch {:3d}, value {:.6f}".format(batch, value))

    return value


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    seed = 4321

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dim", type=int)
    parser.add_argument("--kernel", type=str)
    parser.add_argument("--stddev", type=float)

    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    print(args)

    device = "cuda:0"

    exponent = np.linspace(math.log10(2 * args.dim), 4, num=10)
    lst_batch = np.rint(10 ** exponent).astype(int)
    lst_error = []

    for batch in lst_batch:
        error = evaluate_error_functions(
            batch,
            d=args.dim, kernel=args.kernel, noise=args.stddev ** 2,
            device=device,
        )
        lst_error.append(error)

    print(lst_error)
    torch.save(
        {
            'lst_batch': lst_batch,
            'lst_error': lst_error,
        }, args.output
    )
