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

`experiments/real_data_gp_maps.py` renders the stored posterior mean maps and
measurement locations. `experiments/real_data_als_offline_replay.py` can run
the formal optimization policies on the measured catalogue as a pipeline
test. That replay maximizes the provisional scalar response; it is not the
paper's variance-based mapping task. A formal real-data study should report
mapping error or uncertainty reduction as well as optimization regret.

`experiments/real_data_als_mapping_replay.py` is the corresponding mapping
smoke test. It compares maximum posterior variance with random sampling on the
measured 57-by-57 grid and reports both truth-aware map RMSE and an online
posterior-uncertainty diagnostic. It uses the published final ALS length scales
as fixed values and therefore validates the replay structure, not a full
reproduction with periodic hyperparameter fitting.

## Completed provisional replay results

The five-seed mapping replay used 25 paired initial points and 99 total
measurements. At the final budget, maximum-variance sampling achieved a mean
normalized map RMSE of 0.2860, compared with 0.2995 for random sampling. Its
mean maximum posterior standard deviation divided by the response range was
also lower: 0.2294 versus 0.3211. No run reached the provisional 0.05
uncertainty threshold. This supports the expected coverage benefit of variance
sampling, but the result is a pipeline smoke because the response is still the
raw-maximum q proxy and the published final length scales are held fixed.

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

1. Is the primary beamline task mapping, optimization, or a staged combination?
2. Which fitted peak and q range define the ALS response?
3. What objective-unit error is scientifically acceptable?
4. Which false-stop probability is acceptable at the beamline?
5. Should the replay use exact grid lookup or nearest-neighbor lookup for
   off-grid proposals?
6. Should measurement time include motor travel and data-reduction latency?

