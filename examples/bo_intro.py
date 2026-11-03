"""One-dimensional closed-loop Bayesian optimization with BoTorch.

The dense objective curve is for plotting only. The optimizer receives only
values returned by objective at selected points.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import botorch
import gpytorch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

DTYPE = torch.double
BOUNDS = torch.tensor([[0.0], [1.0]], dtype=DTYPE)


def objective(x: torch.Tensor) -> torch.Tensor:
    """Toy maximization objective; x has shape (..., 1)."""
    return torch.sin(6 * torch.pi * x) + 0.4 * x


def fit_model(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def propose(model: SingleTaskGP, best_f: float) -> tuple[torch.Tensor, float]:
    acq = LogExpectedImprovement(model=model, best_f=best_f)
    x_next, log_ei = optimize_acqf(
        acq, bounds=BOUNDS, q=1, num_restarts=12, raw_samples=256
    )
    return x_next.detach(), float(log_ei.detach())


def run_bo(seed: int = 17, n_steps: int = 6):
    torch.manual_seed(seed)
    train_x = torch.tensor([[0.08], [0.37], [0.72]], dtype=DTYPE)
    train_y = objective(train_x)
    history = []

    for step in range(n_steps):
        model = fit_model(train_x, train_y)
        best_before = float(train_y.max())
        x_next, log_ei = propose(model, best_before)
        y_next = objective(x_next)  # one new expensive evaluation
        history.append(
            {
                "step": step + 1,
                "x_next": x_next.item(),
                "y_next": y_next.item(),
                "best_before": best_before,
                "best_after": max(best_before, y_next.item()),
                "log_ei": log_ei,
            }
        )
        train_x = torch.cat([train_x, x_next])
        train_y = torch.cat([train_y, y_next])

    return train_x, train_y, fit_model(train_x, train_y), history


def make_figure(train_x, train_y, model, output_path: Path):
    grid = torch.linspace(0, 1, 600, dtype=DTYPE).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(grid)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid.squeeze(), objective(grid).squeeze(), "k--", label="true objective")
    ax.plot(grid.squeeze(), mean, color="#2364aa", label="GP posterior mean")
    ax.fill_between(
        grid.squeeze(), mean - 2 * std, mean + 2 * std,
        color="#2364aa", alpha=0.2, label="latent mean ± 2 SD"
    )
    ax.scatter(train_x.squeeze(), train_y.squeeze(), color="#d1495b", zorder=3,
               label="evaluated points")
    ax.set(xlabel="x", ylabel="f(x)", title="Closed-loop Bayesian optimization")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return fig


def make_progress_figure(history, output_path: Path):
    steps = [0] + [row["step"] for row in history]
    best = [history[0]["best_before"]] + [row["best_after"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.step(steps, best, where="post", color="#2364aa", linewidth=2)
    ax.scatter(steps, best, color="#2364aa", zorder=3)
    ax.set(
        xlabel="BO iteration after the initial design",
        ylabel="best observed objective",
        title="Best-so-far value is monotone, but flat steps do not prove completion",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return fig


def save_results(train_x, train_y, history, result_dir: Path, seed: int = 17):
    if not history:
        raise ValueError("history must contain at least one BO iteration")
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "bo_intro_trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    grid = torch.linspace(0, 1, 20001, dtype=DTYPE).unsqueeze(-1)
    true_values = objective(grid).squeeze(-1)
    true_idx = int(true_values.argmax())
    rec_idx = int(train_y.squeeze(-1).argmax())
    summary = {
        "seed": seed,
        "initial_observations": 3,
        "bo_iterations": len(history),
        "objective_evaluations": int(train_x.shape[0]),
        "recommendation_x": float(train_x[rec_idx]),
        "recommendation_y": float(train_y[rec_idx]),
        "dense_reference_x": float(grid[true_idx]),
        "dense_reference_y": float(true_values[true_idx]),
        "dense_reference_simple_regret": float(true_values[true_idx] - train_y[rec_idx]),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "botorch": botorch.__version__,
            "gpytorch": gpytorch.__version__,
        },
        "note": "The dense reference grid is used only for retrospective teaching and scoring, never to choose x_next.",
    }
    (result_dir / "bo_intro_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    x, y, fitted_model, trace = run_bo()
    for row in trace:
        print(
            f"step={row['step']:02d} x_next={row['x_next']:.5f} "
            f"y={row['y_next']:.5f} best_after={row['best_after']:.5f}"
        )
    make_figure(x, y, fitted_model, Path("assets/bo_intro_result.png"))
    make_progress_figure(trace, Path("assets/bo_intro_progress.png"))
    summary = save_results(x, y, trace, Path("results/phase1"))
    print(json.dumps(summary, indent=2))

