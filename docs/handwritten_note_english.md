# Digitized English Notes

## Core idea

We do not know the relationship between an experimental input `x` and outcome
`y`. We can afford only a small number of experiments. A Gaussian process uses
the observed `(x, y)` pairs to represent plausible unknown functions and gives
a posterior mean and uncertainty at every candidate input.

An acquisition function converts those two outputs into a value for each
candidate. Maximizing the acquisition gives `x_next`. After observing
`y_next`, we update the dataset and refit the GP. The loop continues until a
stopping criterion is met, and then a recommendation rule returns `x_hat`.

## Terms that must not be mixed up

- `f(x)`: the real but unknown objective.
- GP posterior: beliefs about `f` after seeing data.
- Kernel: assumptions about similarity and smoothness.
- Acquisition: value of sampling a candidate next.
- Acquisition optimizer: numerical routine that maximizes the acquisition.
- Stopping rule: test for ending data collection.
- Recommendation rule: rule for selecting the final `x`.

## Hyperparameters

- Length scale controls horizontal smoothness.
- Output scale controls vertical variation.
- Noise controls how much disagreement is treated as measurement error.
- Mean function sets the prior baseline.

Hyperparameters are fitted to the observations, usually by marginal likelihood.
They can be re-estimated after each new observation. Re-fitting does not make a
wrong kernel correct; diagnostics and adversarial benchmark functions are part
of this project's scientific contribution.

## Key warning

Low uncertainty is conditional on the GP model. If the kernel assumes a smooth
landscape but the objective contains a narrow peak, the model may assign low
uncertainty to a region it has misunderstood. Therefore stopping performance
must be evaluated using both easy matched cases and deliberately misspecified
cases.

