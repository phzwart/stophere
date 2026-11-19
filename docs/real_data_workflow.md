# Real scattering data workflow

## Source and privacy boundary

Peter provided a private peer-review share of the data associated with
*Toward Unified Autonomous Scattering Experiments: A Cross-Facility Case Study
at ALS and PETRA III*. The share token, downloaded archives, extracted HDF5
files, and data-derived private outputs must not be committed. This repository
contains only reusable readers, audits, and provisional replay code.

The processed ALS and PETRA III archives can be read with `h5py`. Use
`experiments/real_data_inventory.py` to record file counts, HDF5 schemas,
readability, duplicate paths, finite-value checks, and largest files. The local
audit found no unreadable HDF5 files, duplicate relative paths, or nonfinite
values in the numeric datasets that were scanned.

## What the real data represent

- ALS beamline 7.3.3 measured dried sodium chloride on Kapton tape with WAXS.
  The sample coordinates are two motor positions. The published output is the
  q-space position of the highest-intensity fitted peak.
- PETRA III P03 measured an aluminum/polyamide composite. Its published output
  is a beamstop-diode absorption signal.
- Both experiments used an anisotropic stationary Matérn GP and variance
  acquisition because the scientific task was spatial mapping, not only
  maximization.
- The final stored GP runs contain 99 ALS measurements and 1,000 PETRA III
  measurements.

## Offline replay data

The processed ALS archive contains 3,249 reduced grid spectra on a complete
57-by-57 motor grid. This is sufficient for nearest-neighbor offline replay:
an algorithm proposes a motor position, the simulator returns the closest
measured spectrum, and the new response is added to the model. Raw detector
frames are not required for this first replay because the reduced `q`,
`intensity`, and `errors` arrays are already present.

`experiments/real_data_als_catalogue.py` builds a local finite catalogue. Its
default response is the q coordinate at the raw maximum of each reduced
profile. This is deliberately labeled a proxy. It validates data ingestion but
does not replace the paper's fitted-peak extraction. Peter should confirm the
peak model, q window, preprocessing, and scientific tolerance before the
result is treated as a research comparison.

`experiments/real_data_als_offline_replay.py` is an earlier one-seed
optimization pipeline check using a provisional scalar response. The current
optimization study is documented in `als_three_target_replay.md`.

## Historical side experiment: mapping

An earlier mapping smoke test compared maximum posterior variance with random
sampling on the measured grid. It is retained as historical work only. Mapping
is outside the current project priority, which is optimization.

`experiments/real_data_als_optimization_targets.py` adds three provisional
optimization responses requested in the November discussion: total integrated
intensity, maximum detected peak height, and minimum peak FWHM. The paired
five-seed replay in `experiments/real_data_als_three_target_replay.py` compares
the five formal BoTorch policies with a random-search reference while keeping
the measured catalogue, Sobol initial design, budget, surrogate, and scoring
rules fixed. The reusable implementation and configuration are committed; the
private extracted catalogue, trajectories, numerical results, and figures are
kept outside the repository. See [the three-target replay note](als_three_target_replay.md).

## Completed provisional replay results

The historical five-seed mapping replay is not part of the current optimization
results or next-step plan.

A one-seed optimization replay also ran LogEI, LogPI, UCB, Thompson sampling,
and MES from the same 25 measured starting points to a total budget of 99.
LogEI and UCB ended at normalized regret 0.0060, MES found the catalogue best
point, LogPI ended at 0.1821, and Thompson sampling at 0.1841. Only the
stagnation observer fired early, at evaluations 45-60; the other scientific
observers did not fire. This is evidence that the complete BoTorch and stopping
pipeline can consume measured scattering data, not a policy ranking: one seed
cannot measure variability, and maximizing the provisional q proxy is not yet
the confirmed beamline objective.

## Remaining scientific decisions

1. Which fitted peak and q range define the ALS optimization response?
2. What objective-unit error is scientifically acceptable for each target?
3. Which false-stop probability is acceptable at the beamline?
4. Should the replay use exact grid lookup or nearest-neighbor lookup for
   off-grid proposals?
5. Should measurement time include motor travel and data-reduction latency?
6. Should integrated intensity use the full q range, a background-subtracted
   range, or a peak-specific window?
7. Should FWHM be measured at half prominence, half raw height, or from a
   fitted peak model?

