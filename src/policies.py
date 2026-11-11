"""Policy-specific candidate selection for the formal BoTorch Phase 3 study."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.generation.sampling import MaxPosteriorSampling


POLICIES = ("log_ei", "log_pi", "ucb", "thompson_sampling", "mes")


@dataclass(frozen=True)
class PolicySettings:
    log_pi_margin: float = 0.0
    ucb_beta: float = 2.0
    mes_num_mv_samples: int = 10
    mes_num_y_samples: int = 128


def _analytic_argmax(acquisition, candidates: torch.Tensor) -> int:
    """Return the local row index maximizing a q=1 acquisition."""
    with torch.no_grad():
        values = acquisition(candidates.unsqueeze(-2)).reshape(-1)
    return int(torch.argmax(values).item())


def select_candidate(
    *,
    policy: str,
    model,
    train_y: torch.Tensor,
    available_candidates: torch.Tensor,
    full_catalogue: torch.Tensor,
    settings: PolicySettings,
    sampling_seed: int,
) -> int:
    """Return a row index into ``available_candidates``.

    The caller owns the common BO loop. This function contains the only logic
    that changes between acquisition policies.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if available_candidates.ndim != 2 or available_candidates.shape[0] == 0:
        raise ValueError("available_candidates must be a non-empty n-by-d tensor")

    best_f = train_y.max()
    if policy == "log_ei":
        return _analytic_argmax(
            LogExpectedImprovement(model=model, best_f=best_f),
            available_candidates,
        )
    if policy == "log_pi":
        return _analytic_argmax(
            LogProbabilityOfImprovement(
                model=model, best_f=best_f + settings.log_pi_margin
            ),
            available_candidates,
        )
    if policy == "ucb":
        return _analytic_argmax(
            UpperConfidenceBound(model=model, beta=settings.ucb_beta),
            available_candidates,
        )
    if policy == "thompson_sampling":
        # MaxPosteriorSampling draws jointly over the candidate set, preserving
        # posterior correlation. fork_rng keeps the stream local to this policy.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(sampling_seed)
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            selected = sampler(available_candidates, num_samples=1).reshape(1, -1)
        distances = torch.sum((available_candidates - selected) ** 2, dim=-1)
        return int(torch.argmin(distances).item())

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sampling_seed)
        acquisition = qMaxValueEntropy(
            model=model,
            candidate_set=full_catalogue,
            num_mv_samples=settings.mes_num_mv_samples,
            num_y_samples=settings.mes_num_y_samples,
        )
        return _analytic_argmax(acquisition, available_candidates)

