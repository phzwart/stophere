"""Visualize how the GP length scale changes extrapolation and uncertainty."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel

from bo_intro import DTYPE, objective


def fixed_lengthscale_model(x, y, lengthscale):
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=1,
                                      lengthscale_constraint=GreaterThan(1e-4)))
    model = SingleTaskGP(x, y, covar_module=kernel)
    model.covar_module.base_kernel.lengthscale = lengthscale
    model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
    model.eval()
    return model


def main():
    x = torch.tensor([[0.08], [0.37], [0.72]], dtype=DTYPE)
    y = objective(x)
    grid = torch.linspace(0, 1, 600, dtype=DTYPE).unsqueeze(-1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)

    for ax, ell in zip(axes, [0.04, 0.15, 0.60]):
        model = fixed_lengthscale_model(x, y, ell)
        with torch.no_grad():
            post = model.posterior(grid)
            mean = post.mean.squeeze(-1)
            std = post.variance.sqrt().squeeze(-1)
        ax.plot(grid.squeeze(), objective(grid).squeeze(), "k--", lw=1)
        ax.plot(grid.squeeze(), mean, color="#2364aa")
        ax.fill_between(grid.squeeze(), mean - 2 * std, mean + 2 * std,
                        color="#2364aa", alpha=.2)
        ax.scatter(x.squeeze(), y.squeeze(), color="#d1495b", zorder=3)
        ax.set_title(f"length scale = {ell}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    fig.suptitle("Same data, different smoothness assumptions")
    fig.tight_layout()
    out = Path("assets/lengthscale_sensitivity.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)


if __name__ == "__main__":
    main()

