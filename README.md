# The Behavior and Convergence of Local Bayesian Optimization

This repository contains the implementation for the paper [The Behavior and Convergence of Local Bayesian Optimization](https://arxiv.org/abs/2305.15349).

## Instructions for Running the Code
The jupyter notebooks `l2_vs_l1_norm.ipynb` and `relu.ipynb` reproduce the experiments on gradient estimation.

The python scripts `evaluate_error_function_vs_batch.py` and `evaluate_error_function_vs_dim.py` reproduce the experiment on the tightness of the bounds on the error function.
The following commands run the python scripts:
```
python evaluate_error_function_vs_batch.py --dim 10 --kernel matern --stddev 0.1 --output ./output1.tar
python evaluate_error_function_vs_dim.py --n 500 --kernel matern --stddev 0.1 --output ./output2.tar
```

Dependency:
- PyTorch 2.0
- GPyTorch 1.9.1
