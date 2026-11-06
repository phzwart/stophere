# Phase 3 extended-budget success profiles

## Protocol

This focused screen extends four existing policies to 150 evaluations on all nine benchmark systems: nominal RBF/UCB, expected improvement, probability of improvement, and random search. Each policy-system pair uses five paired seeds. Success is normalized simple regret at most 0.05.

The experiment contains 180 trajectories. Success is reported at 5, 10, 20, 50, 100, and 150 evaluations. The pooled rate treats every trajectory equally. The 25th percentile, median, and 75th percentile are calculated from the nine system-level success rates; they describe variation across systems and are not confidence intervals.

## Results

| Policy | Success at 50 | Success at 100 | Success at 150 |
| --- | ---: | ---: | ---: |
| Nominal RBF/UCB | 77.8% | 80.0% | 80.0% |
| Expected improvement | 77.8% | 84.4% | 86.7% |
| Probability of improvement | 77.8% | 77.8% | 80.0% |
| Random search | 55.6% | 66.7% | 66.7% |

All three BO policies reach 77.8% success by 50 evaluations. Expected improvement continues to improve and has the highest measured success at 150 evaluations, while nominal UCB and probability of improvement largely plateau. The apparent median system-level success is 100% at 150 evaluations for every method, but the pooled rates are lower because failures are concentrated in a few hard systems. The curved-valley and hidden-narrow systems remain the main BO failure cases; random search also struggles on the flat multimodal and rough-field systems.

These are characterization results, not a final algorithm ranking. Five seeds give coarse system-level rates, and the current screen contains only the existing BO policies plus random search. The full comparison should use ten paired seeds and add validated BAM, SNOBFIT, MCS, CMA-ES, and NOMAD adapters on the continuous box-constrained systems.

## Outputs

- `results/phase3/budget_profiles/trajectories.csv`: step-by-step results for all 180 trajectories.
- `results/phase3/budget_profiles/success_profiles.csv`: pooled rates and system-level quantiles.
- `results/phase3/budget_profiles/success_by_budget.png`: success at each evaluation checkpoint.
- `results/phase3/budget_profiles/success_quantiles_final_budget.png`: final success bars, interquartile ranges, and pooled-rate markers.

