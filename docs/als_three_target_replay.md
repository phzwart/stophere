# ALS three-target offline optimization replay

## Purpose

This workflow turns the processed ALS reduced spectra into a finite measured
catalogue and asks three different optimization questions:

1. maximize total integrated intensity;
2. maximize the height of the strongest detected peak;
3. minimize the full width at half maximum (FWHM) of that peak.

The three targets are intentionally different. Integrated intensity rewards
total scattering signal, peak height rewards a strong local feature, and FWHM
rewards a narrow feature. They need not select the same motor position.

## Data boundary

The source data are private peer-review files associated with the public paper
*Toward Unified Autonomous Scattering Experiments: A Cross-Facility Case Study
at ALS and PETRA III* (dataset DOI: `10.5061/dryad.p2ngf1w4t`). The private
share token, downloaded files, extracted catalogue, trajectories, figures, and
numerical summaries are not committed. This repository contains only reusable
code and provisional configuration.

## Target extraction

`experiments/real_data_als_optimization_targets.py` reads the reduced `q`,
`intensity`, and `errors` arrays at every measured motor location.

- Integrated intensity is the trapezoidal integral of the recorded intensity
  over the common measured q range.
- The peak-height target uses a Savitzky-Golay-smoothed profile and the tallest
  interior peak that exceeds a documented error- and range-based prominence
  threshold.
- FWHM is measured at half prominence for the same peak. A spectrum without a
  valid peak receives a documented worst-value penalty.

These extraction choices are scientifically plausible but provisional. The
q window, background treatment, smoothing, peak definition, and width
definition require collaborator approval before publication claims.

## Fair replay protocol

`experiments/real_data_als_three_target_replay.py` compares LogEI, LogPI, UCB,
Thompson sampling, MES, and a random-search reference on the same finite
catalogue. Every paired seed uses the same 25 Sobol initial locations, the same
99-measurement budget, the same Matérn-5/2 ARD Gaussian process, the same
recommendation rule, and the same provisional success definition. No
synthetic measurement noise is added. A small declared observation-noise term
is used for numerical regularization.

The Phase 4 stopping rules are passive observers: they see only information
available at the current evaluation, record their first trigger, and never
change the acquisition trajectory. Truth-aware regret and false-stop scoring
are computed only after the replay.

## How to run locally

First build the private target catalogue, then run the paired replay. Paths to
the private input and private output directory must be supplied locally.

~~~bash
python -m experiments.real_data_als_optimization_targets PRIVATE_ALS_DIRECTORY \
  --output PRIVATE_OUTPUT_DIRECTORY

python -m experiments.real_data_als_three_target_replay \
  PRIVATE_OUTPUT_DIRECTORY/als_optimization_targets.csv \
  --output PRIVATE_OUTPUT_DIRECTORY/full_replay_regularized \
  --seeds 5 --workers 5
~~~

The replay is resumable: completed target-policy-seed trajectories are reused.

## Interpretation limits

This is an offline replay over real measured spectra, not a new autonomous
beamline run. The measurement values and motor coordinates are real; the
sequence in which the algorithm queries them is simulated. The catalogue
optimum is available for retrospective scoring but is never exposed to the
acquisition policies or stopping observers. Five seeds are enough for a
pre-pilot, not for a final uncertainty statement.

Before freezing the scientific protocol, confirm the primary objective, q
window, preprocessing, acceptable objective-unit error, false-stop risk,
measurement budget, and whether travel time should enter the cost model.
