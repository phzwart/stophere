"""Shared finite-catalogue BoTorch runner for formal Phase 3 comparisons."""
from __future__ import annotations

import csv
import json
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import botorch
import gpytorch
import numpy as np
import torch
import yaml
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from src.benchmarks import Benchmark
from src.policies import POLICIES, PolicySettings, select_candidate
from src.stopping import RULES, StoppingProtocol, evaluate_stopping_observers


@dataclass(frozen=True)
class Protocol:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Protocol":
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw["domain"]["mode"] != "finite_catalogue":
            raise ValueError("this runner implements the finite-catalogue protocol")
        return cls(raw=raw)

    def initial_size(self, dimension: int) -> int:
        return int(self.raw["initial_design"]["points_per_dimension"]) * dimension

    def budget(self, dimension: int) -> int:
        return int(self.raw["budget"][f"dimension_{dimension}"])

    @property
    def catalogue_size(self) -> int:
        return int(self.raw["domain"]["catalogue_size"])

    @property
    def policy_settings(self) -> PolicySettings:
        values = self.raw["acquisition"]
        return PolicySettings(
            log_pi_margin=float(values["log_pi_margin"]),
            ucb_beta=float(values["ucb_beta"]),
            mes_num_mv_samples=int(values["mes_num_mv_samples"]),
            mes_num_y_samples=int(values["mes_num_y_samples"]),
        )


def make_catalogue(dimension: int, protocol: Protocol) -> torch.Tensor:
    seed = int(protocol.raw["domain"]["catalogue_seed_base"]) + dimension
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    return engine.draw(protocol.catalogue_size, dtype=torch.double)


def initial_indices(
    *, dimension: int, optimizer_seed: int, protocol: Protocol
) -> np.ndarray:
    seed = int(protocol.raw["initial_design"]["seed_base"]) + optimizer_seed
    rng = np.random.default_rng(seed)
    return rng.choice(
        protocol.catalogue_size,
        size=protocol.initial_size(dimension),
        replace=False,
    )


def _fit_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    protocol: Protocol,
    model_seed: int,
):
    settings = protocol.raw["surrogate"]
    dimension = train_x.shape[-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        covariance = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=dimension,
                lengthscale_prior=GammaPrior(
                    float(settings["lengthscale_prior_shape"]),
                    float(settings["lengthscale_prior_rate"]),
                ),
            ),
            outputscale_prior=GammaPrior(
                float(settings["outputscale_prior_shape"]),
                float(settings["outputscale_prior_rate"]),
            ),
        )
        train_yvar = torch.full_like(
            train_y, float(settings["numerical_noise_variance"])
        )
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            train_Yvar=train_yvar,
            covar_module=covariance,
            outcome_transform=Standardize(m=1),
        )
        marginal_likelihood = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_gpytorch_mll(marginal_likelihood)
    model.eval()
    return model, [str(item.message) for item in caught]


def _model_parameters(model) -> dict[str, Any]:
    kernel = model.covar_module
    return {
        "lengthscales": [
            float(value) for value in kernel.base_kernel.lengthscale.detach().reshape(-1)
        ],
        "outputscale": float(kernel.outputscale.detach().item()),
    }


def run_trajectory(
    *,
    benchmark: Benchmark,
    policy: str,
    optimizer_seed: int,
    protocol: Protocol,
    catalogue: torch.Tensor | None = None,
    budget_override: int | None = None,
    stopping_protocol: StoppingProtocol | None = None,
) -> list[dict[str, Any]]:
    """Run one policy to the hard budget and return a row per observation count."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if benchmark.dimension not in (2, 3):
        raise ValueError("formal continuous Phase 3 currently supports 2D and 3D")

    candidates = (
        make_catalogue(benchmark.dimension, protocol)
        if catalogue is None
        else catalogue
    )
    if candidates.shape != (protocol.catalogue_size, benchmark.dimension):
        raise ValueError("catalogue shape does not match the frozen protocol")
    objective_values = torch.as_tensor(
        benchmark(candidates.detach().cpu().numpy()), dtype=torch.double
    ).reshape(-1, 1)
    reference_value = float(objective_values.max().item())
    value_range = max(
        float((objective_values.max() - objective_values.min()).item()), 1e-12
    )

    selected = initial_indices(
        dimension=benchmark.dimension,
        optimizer_seed=optimizer_seed,
        protocol=protocol,
    ).tolist()
    used = torch.zeros(protocol.catalogue_size, dtype=torch.bool)
    used[selected] = True
    budget = (
        protocol.budget(benchmark.dimension)
        if budget_override is None
        else int(budget_override)
    )
    if budget < len(selected) or budget > protocol.catalogue_size:
        raise ValueError("budget must include initialization and fit in the catalogue")

    rows: list[dict[str, Any]] = []
    incumbent_history: list[float] = []
    first_triggers: dict[str, int | None] = {rule: None for rule in RULES}
    sampling_base = int(protocol.raw["acquisition"]["thompson_seed_base"])
    model_seed_base = int(protocol.raw["seeds"]["model_fit_seed_base"])
    stopping_seed_base = (
        int(stopping_protocol.raw["monte_carlo"]["seed_base"])
        if stopping_protocol is not None
        else 0
    )
    while True:
        train_x = candidates[selected]
        train_y = objective_values[selected]
        model, fit_warnings = _fit_model(
            train_x,
            train_y,
            protocol,
            model_seed=model_seed_base + optimizer_seed * 1000 + len(selected),
        )
        incumbent_local = int(torch.argmax(train_y).item())
        incumbent_index = int(selected[incumbent_local])
        incumbent_value = float(train_y[incumbent_local].item())
        incumbent_history.append(incumbent_value)
        regret = max(reference_value - incumbent_value, 0.0)
        parameters = _model_parameters(model)
        row = {
            "system": benchmark.name,
            "dimension": benchmark.dimension,
            "policy": policy,
            "seed": optimizer_seed,
            "n_eval": len(selected),
            "last_selected_index": int(selected[-1]),
            "selected_indices": json.dumps([int(index) for index in selected]),
            "incumbent_index": incumbent_index,
            "incumbent_value": incumbent_value,
            "reference_value": reference_value,
            "simple_regret": regret,
            "normalized_regret": regret / value_range,
            "success_le_0.05": int(regret / value_range <= 0.05),
            "lengthscales": json.dumps(parameters["lengthscales"]),
            "outputscale": parameters["outputscale"],
            "fit_warning_count": len(fit_warnings),
            "fit_warnings": " | ".join(fit_warnings),
        }
        if stopping_protocol is not None:
            stopping = evaluate_stopping_observers(
                protocol=stopping_protocol,
                model=model,
                train_y=train_y,
                catalogue=candidates,
                recommendation_index=incumbent_index,
                incumbent_history=incumbent_history,
                n_eval=len(selected),
                initial_size=protocol.initial_size(benchmark.dimension),
                hard_budget=budget,
                sampling_seed=(
                    stopping_seed_base
                    + optimizer_seed * 1000
                    + len(selected)
                ),
            )
            row.update(stopping)
            for rule in RULES:
                if stopping[f"{rule}_fire"] and first_triggers[rule] is None:
                    first_triggers[rule] = len(selected)
                row[f"{rule}_first_trigger"] = first_triggers[rule]
        rows.append(row)
        if len(selected) == budget:
            break

        available_global = torch.nonzero(~used, as_tuple=False).reshape(-1)
        available = candidates[available_global]
        local_index = select_candidate(
            policy=policy,
            model=model,
            train_y=train_y,
            available_candidates=available,
            full_catalogue=candidates,
            settings=protocol.policy_settings,
            sampling_seed=sampling_base + optimizer_seed * 1000 + len(selected),
        )
        global_index = int(available_global[local_index].item())
        if used[global_index]:
            raise RuntimeError("policy selected an already evaluated candidate")
        used[global_index] = True
        selected.append(global_index)
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "gpytorch": gpytorch.__version__,
        "numpy": np.__version__,
    }

