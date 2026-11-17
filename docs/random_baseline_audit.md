# Random-baseline audit

The extended-budget random baseline samples unevaluated catalogue points
uniformly without replacement. It does not repeat points and it does not stop
early. Therefore, an apparent success plateau is not an implementation bug.

For a catalogue with `N` points, `K` successful points, and a budget of `b`,
the exact probability that random search has seen at least one successful point
is

`1 - choose(N - K, b) / choose(N, b)`.

`experiments/phase3_random_baseline_audit.py` reconstructs `K` from every
benchmark catalogue under the declared normalized-regret threshold and compares
that exact expectation with the five observed paired seeds. Across the eight
continuous systems, the exact mean success probability is 21.1% at 5
evaluations, 48.7% at 50, 59.5% at 100, and 66.8% at 150. The corresponding
observed values are 22.5%, 55.0%, 62.5%, and 62.5%.

The observed 100-to-150 plateau is therefore compatible with coarse five-seed
sampling. The exact expectation continues to increase. Systems with only a
small fraction of successful catalogue points remain difficult for uninformed
sampling even at 150 evaluations.

Outputs:

- `results/phase3/random_baseline_audit/system_success_probabilities.csv`
- `results/phase3/random_baseline_audit/pooled_success_by_budget.csv`
- `results/phase3/random_baseline_audit/random_baseline_audit.png`

