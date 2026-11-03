# Generated Phase 1 figures

These figures are reproducible outputs, not hand-edited illustrations.

- `bo_intro_progress.png`: all six BO iterations in one overview.
- `bo_intro_result.png`: final GP posterior, acquisition decision, and best-so-far trace.
- `lengthscale_sensitivity.png`: how short and long GP length scales change interpolation and uncertainty.

Regenerate them from the repository root with:

```bash
MPLBACKEND=Agg python examples/bo_intro.py
MPLBACKEND=Agg python examples/02_lengthscale_sensitivity.py
```
