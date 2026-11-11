# Translating the stopping study to a beamline experiment

## What is known and what remains open

The current computational project does not yet identify a specific beamline,
control vector, detector signal, or scientific sample. Those choices belong in
the proposed group meeting. Until they are fixed, the repository should not
claim that the live experiment will optimize protein diffraction, beam power
density, or any other particular observable.

The most direct translation is an autonomous-alignment experiment at an X-ray
beamline. Motors or other optical settings become the BO input vector. A detector
returns beam-quality measurements such as intensity, position, spot size, shape,
or a derived power-density score. Bayesian optimization proposes the next safe
setting, and the stopping observer decides when additional measurements are no
longer worth their cost.

At a macromolecular-crystallography beamline, an alternative experiment could
use a crystal sample and an area detector. The measured X-ray diffraction pattern
contains spots whose positions and intensities support structure determination.
Possible optimization objectives include diffraction strength, resolution,
indexing success, signal-to-noise, or a quality-versus-radiation-damage trade-off.
This is a possible application, not the currently frozen experimental target.

## Why the beamline data matter

Synthetic benchmarks provide known truth, controlled difficulty, and cheap
repetition. A live beamline introduces effects the benchmark cannot fully mimic:

- motor backlash, drift, calibration error, and constraints;
- detector noise, saturation, and failed readings;
- nonstationary behavior as the instrument warms or the sample changes;
- real evaluation latency and operational risk;
- unknown ground truth, so stopping cannot use the true optimum.

The live experiment tests whether the acquisition and stopping design remains
useful under those conditions. Its value is not merely finding a high score; it
is reducing setup or measurement time while preserving beam or data quality and
maintaining safe operating limits.

## Required meeting decisions

1. Select the beamline and responsible beamline scientist.
2. Define the controllable variables, bounds, units, and safe initial state.
3. Define the detector outputs and the scalar or multi-objective quality score.
4. Set abort conditions, forbidden regions, rollback behavior, and operator
   override.
5. Measure evaluation latency and decide the maximum live evaluation budget.
6. Choose a conservative initial design and a manual or scripted baseline.
7. Specify what the algorithm may log and how failed or low-fidelity readings are
   handled.
8. Begin with recommendation-only operation, then supervised closed-loop control,
   before considering autonomous operation.

## Relevant public examples

- [ALS autonomous beamline overview](https://als.lbl.gov/autonomous-beamlines-harnessing-bayesian-optimization-to-revolutionize-accelerator-science/)
- [General Bayesian algorithm for autonomous beamline alignment](https://pmc.ncbi.nlm.nih.gov/articles/PMC11542659/)
- [ALS-ENABLE beamline capabilities](https://als-enable.lbl.gov/wordpress/our-beamlines/)

