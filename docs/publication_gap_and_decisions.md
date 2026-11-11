# Publication gap, flexible research directions, and collaborator decisions

## Current position relative to the literature

The project is not currently a new acquisition-function paper. LogEI, LogPI,
UCB, Thompson sampling, and MES are established methods. It is also not yet a
new stopping-rule paper: expected-regret, probabilistic-regret, and cost-aware
rules already exist. The present contribution is a controlled characterization
framework that keeps acquisition, recommendation, stopping, and truth-based
evaluation separate.

The closest stopping papers each emphasize one axis:

- Ishibashi et al. study expected simple-regret change;
- Wilson studies posterior `(epsilon, delta)` probabilistic regret;
- Xie et al. study cost-aware stopping coupled to cost-aware acquisition;
- Makarova et al. study termination for hyperparameter optimization.

The closest beamline work demonstrates autonomous alignment, low-fidelity data
handling, or simulation-to-real transfer, but does not provide a systematic
calibration study of stopping reliability across acquisition policies.

The defensible gap is therefore not merely a 5-by-5 table. It is the question:

> When an acquisition policy changes the data that a GP sees, when do common
> stopping claims become overconfident, and can a stopping decision be
> calibrated to the cost and failure modes of an autonomous beamline?

## Research directions that remain flexible

1. **Calibration under model misspecification.** Compare claimed posterior
   stop probability with empirical success, stratified by kernel mismatch,
   length scale, noise, narrow peaks, and invalid observations.
2. **Cost-aware beamline stopping.** Replace a uniform evaluation count with
   exposure time, motor travel, settling time, dose, and risk of losing the
   beam. This must be defined with beamline staff rather than guessed.
3. **Acquisition-stopping interaction.** Use the same stopping observer on
   trajectories from all five acquisitions and identify which policies create
   deceptively confident posteriors.
4. **Selective or abstaining stops.** Permit a rule to say that its assumptions
   are unsupported and fall back to the hard budget or request a diagnostic
   measurement.
5. **Invalid-data and drift robustness.** Add detector clipping, beam loss,
   background contamination, heteroscedastic noise, and slow drift as focused
   confirmation experiments.
6. **Sim-to-real confirmation.** Develop on a digital twin or replayable public
   catalogue, then confirm only selected conditions online. Offline replay
   cannot evaluate arbitrary counterfactual queries unless the domain is a
   complete catalogue or a trusted simulator.

Do not claim any item is the first of its kind until the final literature
review is complete.

## Work that can proceed before beamline selection

- finish the formal Phase 3 smoke and pre-pilot under the finite-catalogue
  protocol;
- validate all five Phase 4 observers without changing acquisition paths;
- build the Phase 5 run manifest, failure accounting, and resumable outputs;
- implement Phase 6 evaluator metrics and uncertainty intervals;
- run synthetic misspecification, noise, and catalogue-sensitivity studies;
- screen public data for format and replay suitability without declaring it a
  substitute for an online beamline experiment.

## Decisions required from Peter before the full matrix

### Required to freeze Phase 3

1. Is the finite catalogue the primary controlled experiment, with continuous
   optimization reserved for confirmation?
2. Are 2,048 candidates, `5*d` initial points, and budgets of 100/150 acceptable?
3. Is normalized simple regret at most 0.05 the primary evaluator success
   definition, with 0.01 and 0.02 as sensitivity thresholds?
4. Is the noiseless matrix primary, with noise kept as a separate extension?
5. Is best evaluated point the noiseless recommendation rule?
6. Should random search remain the main reference and CMA-ES, SNOBFIT, NOMAD,
   and BAM remain a separate continuous-domain comparison until domains match?

### Required to freeze Phase 4

7. What does an acceptable error mean in scientific units? The current
   `epsilon=0.02` is provisional and is not meaningful for every unscaled task.
8. What false-stop risk is acceptable: 5%, 1%, or a cost-weighted decision?
9. Should repeated checks be family-wise controlled, or should the project
   report conditional posterior probabilities without calling them guarantees?
10. Should cost be a secondary evaluation metric or part of the stopping rule?

### Required for public data and Phase 7

11. Which facility, beamline, instrument, and staff collaborator are plausible?
12. Which controls are decision variables: mirror pitch/roll, translations,
    apertures, monochromator settings, sample position, or another set?
13. Which measured output defines success: flux, spot size, centroid stability,
    spectral resolution, signal-to-noise, or a multi-objective combination?
14. What are safe bounds, forbidden moves, exposure times, settling times, and
    beam-loss conditions?
15. Does a digital twin, historical scan, Bluesky run, or open dataset exist,
    and does it contain input settings paired with raw and reduced outcomes?
16. Can historical data support only surrogate testing, or is the candidate
    catalogue dense enough for offline policy replay?
17. What approvals, beamtime proposal, data policy, and operator supervision
    are required before online optimization?

## Public-data rule

Public beamline data are useful for testing ingestion, surrogate fitting,
noise models, detector-quality filters, and offline catalogue studies. A sparse
historical log cannot honestly compare arbitrary sequential acquisition
policies, because it does not contain the outcomes of unchosen counterfactual
settings. Prefer, in order:

1. a beamline digital twin or simulator;
2. a dense motor-setting catalogue with outcomes;
3. a historical dataset for surrogate/noise studies only;
4. real online experiments for final confirmation.

Dataset selection should follow the intended beamline and measurement, not the
other way around. A generic X-ray image archive is not automatically an
optimization dataset.
