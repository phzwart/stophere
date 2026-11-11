"""Trajectory-preserving stopping observers for the formal Phase 4 study.

The observers in this module never choose an evaluation point and never read
benchmark truth.  They inspect only the fitted model and observations that are
available at the current iteration.  The acquisition runner remains responsible
for continuing every trajectory to its hard budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from scipy.stats import beta as beta_distribution


RULES = (
    "fixed_budget",
    "incumbent_stagnation",
    "negligible_expected_improvement",
    "global_confidence_gap",
    "posterior_acceptable_regret",
)


@dataclass(frozen=True)
class StoppingProtocol:
    """Versioned Phase 4 observer settings."""

    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "StoppingProtocol":
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls(raw=raw)

    @property
    def epsilon(self) -> float:
        return float(self.raw["thresholds"]["epsilon_objective_units"])

    @property
    def num_paths(self) -> int:
        return int(self.raw["monte_carlo"]["num_paths"])

    def maximum_checks(self, *, initial_size: int, hard_budget: int) -> int:
        start = initial_size + int(
            self.raw["check_schedule"]["start_after_initial_extra"]
        )
        every = int(self.raw["check_schedule"]["every_evaluations"])
        scheduled = 0
        if start <= hard_budget:
            scheduled = 1 + (hard_budget - start) // every
        final_is_scheduled = start <= hard_budget and (hard_budget - start) % every == 0
        return max(scheduled + (0 if final_is_scheduled else 1), 1)

    def should_check(
        self, *, n_eval: int, initial_size: int, hard_budget: int
    ) -> bool:
        if n_eval == hard_budget:
            return True
        start = initial_size + int(
            self.raw["check_schedule"]["start_after_initial_extra"]
        )
        every = int(self.raw["check_schedule"]["every_evaluations"])
        return n_eval >= start and (n_eval - start) % every == 0


def _maximum_expected_improvement(
    *, model, train_y: torch.Tensor, catalogue: torch.Tensor
) -> float:
    acquisition = LogExpectedImprovement(model=model, best_f=train_y.max())
    with torch.no_grad():
        maximum_log_ei = acquisition(catalogue.unsqueeze(-2)).reshape(-1).max()
    # Conversion back to objective units is safe for the declared threshold:
    # underflow can only turn a value far below the threshold into zero.
    return float(torch.exp(maximum_log_ei).item())


def _simultaneous_confidence_gap(
    *,
    model,
    catalogue: torch.Tensor,
    recommendation_index: int,
    total_delta: float,
    maximum_checks: int,
) -> tuple[float, float]:
    """Return a finite-catalogue posterior gap and its Gaussian multiplier.

    A Bonferroni allocation covers two tails, every catalogue point, and every
    scheduled look.  This is a conditional posterior construction, not a
    frequentist guarantee under model misspecification.
    """
    posterior = model.posterior(catalogue, observation_noise=False)
    mean = posterior.mean.reshape(-1)
    standard_deviation = posterior.variance.clamp_min(0.0).sqrt().reshape(-1)
    tail_probability = total_delta / (
        2.0 * catalogue.shape[0] * maximum_checks
    )
    standard_normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=catalogue.dtype, device=catalogue.device),
        torch.tensor(1.0, dtype=catalogue.dtype, device=catalogue.device),
    )
    multiplier = standard_normal.icdf(
        torch.tensor(
            1.0 - tail_probability,
            dtype=catalogue.dtype,
            device=catalogue.device,
        )
    )
    upper = mean + multiplier * standard_deviation
    lower_recommendation = (
        mean[recommendation_index]
        - multiplier * standard_deviation[recommendation_index]
    )
    gap = (upper.max() - lower_recommendation).clamp_min(0.0)
    return float(gap.item()), float(multiplier.item())


def _clopper_pearson_lower(
    *, successes: int, trials: int, family_error: float, maximum_checks: int
) -> float:
    """One-sided binomial lower bound with Bonferroni repeated-look control."""
    if successes <= 0:
        return 0.0
    per_check_error = family_error / maximum_checks
    return float(
        beta_distribution.ppf(
            per_check_error,
            successes,
            trials - successes + 1,
        )
    )


def _posterior_regret_probability(
    *,
    model,
    catalogue: torch.Tensor,
    recommendation_index: int,
    epsilon: float,
    num_paths: int,
    sampling_seed: int,
    mc_family_error: float,
    maximum_checks: int,
) -> tuple[float, float, int]:
    """Estimate P(max f - f(recommendation) <= epsilon | data).

    Matheron paths provide scalable joint draws over the complete catalogue.
    The maximum and recommendation value are computed from the same path.  The
    current BoTorch implementation uses random Fourier feature paths, so this
    remains an explicitly recorded numerical approximation.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sampling_seed)
        paths = draw_matheron_paths(model, sample_shape=torch.Size([num_paths]))
        with torch.no_grad():
            values = paths(catalogue)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    regrets = values.max(dim=-1).values - values[..., recommendation_index]
    successes = int((regrets <= epsilon).sum().item())
    estimate = successes / num_paths
    lower = _clopper_pearson_lower(
        successes=successes,
        trials=num_paths,
        family_error=mc_family_error,
        maximum_checks=maximum_checks,
    )
    return estimate, lower, successes


def evaluate_stopping_observers(
    *,
    protocol: StoppingProtocol,
    model,
    train_y: torch.Tensor,
    catalogue: torch.Tensor,
    recommendation_index: int,
    incumbent_history: list[float],
    n_eval: int,
    initial_size: int,
    hard_budget: int,
    sampling_seed: int,
) -> dict[str, Any]:
    """Evaluate all five observers at one iteration without stopping the run."""
    maximum_checks = protocol.maximum_checks(
        initial_size=initial_size, hard_budget=hard_budget
    )
    checked = protocol.should_check(
        n_eval=n_eval, initial_size=initial_size, hard_budget=hard_budget
    )
    result: dict[str, Any] = {
        "stopping_checked": int(checked),
        "fixed_budget_fire": int(n_eval == hard_budget),
        "incumbent_stagnation_fire": 0,
        "negligible_expected_improvement_fire": 0,
        "global_confidence_gap_fire": 0,
        "posterior_acceptable_regret_fire": 0,
        "stagnation_window_gain": math.nan,
        "maximum_expected_improvement": math.nan,
        "global_confidence_gap": math.nan,
        "global_confidence_multiplier": math.nan,
        "posterior_regret_probability": math.nan,
        "posterior_regret_probability_lower": math.nan,
        "posterior_regret_successes": 0,
        "posterior_regret_paths": 0,
        "maximum_scheduled_checks": maximum_checks,
    }
    if not checked:
        return result

    thresholds = protocol.raw["thresholds"]
    window = int(thresholds["stagnation_window"])
    if len(incumbent_history) > window:
        gain = incumbent_history[-1] - incumbent_history[-1 - window]
        result["stagnation_window_gain"] = gain
        result["incumbent_stagnation_fire"] = int(
            gain <= float(thresholds["stagnation_gain"])
        )

    maximum_ei = _maximum_expected_improvement(
        model=model, train_y=train_y, catalogue=catalogue
    )
    result["maximum_expected_improvement"] = maximum_ei
    result["negligible_expected_improvement_fire"] = int(
        maximum_ei <= float(thresholds["maximum_expected_improvement"])
    )

    confidence = protocol.raw["confidence"]
    gap, multiplier = _simultaneous_confidence_gap(
        model=model,
        catalogue=catalogue,
        recommendation_index=recommendation_index,
        total_delta=float(confidence["global_gap_total_delta"]),
        maximum_checks=maximum_checks,
    )
    result["global_confidence_gap"] = gap
    result["global_confidence_multiplier"] = multiplier
    result["global_confidence_gap_fire"] = int(gap <= protocol.epsilon)

    estimate, lower, successes = _posterior_regret_probability(
        model=model,
        catalogue=catalogue,
        recommendation_index=recommendation_index,
        epsilon=protocol.epsilon,
        num_paths=protocol.num_paths,
        sampling_seed=sampling_seed,
        mc_family_error=float(confidence["mc_family_error"]),
        maximum_checks=maximum_checks,
    )
    result["posterior_regret_probability"] = estimate
    result["posterior_regret_probability_lower"] = lower
    result["posterior_regret_successes"] = successes
    result["posterior_regret_paths"] = protocol.num_paths
    result["posterior_acceptable_regret_fire"] = int(
        lower >= 1.0 - float(confidence["posterior_regret_delta"])
    )
    return result
