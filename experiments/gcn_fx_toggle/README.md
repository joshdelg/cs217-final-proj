# GCN Fairness Harness (Single Script)

Single-script harness to compare three aggregation paths under identical timing boundaries and inputs:

- pure Torch aggregation
- manual NKI kernel insertion
- FX-rewritten NKI kernel insertion

## Script

- `run.py`

## Modes and flags

- `--mode torch`
- `--mode manual_nki`
- `--mode fx_nki`
- `--mode compare` (runs all three; default)

Additional fairness controls:

- `--warmup N` (default `1`)
- `--trials N` (default `5`)

## Run

```bash
python experiments/gcn_fx_toggle/run.py --mode compare --warmup 1 --trials 5
python experiments/gcn_fx_toggle/run.py --mode manual_nki --warmup 1 --trials 5
python experiments/gcn_fx_toggle/run.py --mode fx_nki --warmup 1 --trials 5
```

## What the harness reports

- Per-trial wall-clock for each selected mode.
- Per-mode summary: `mean/std/min/max`.
- Output difference metrics versus Torch baseline (when Torch is included):
  - `max_abs_diff`
  - `mean_abs_diff`
- Speedups between mode means.

## Notes

- Timing covers the same end-to-end 2-layer GCN forward region for every mode.
- Warmup iterations are excluded from timed statistics.
- Prefer `--mode compare` for the fairest side-by-side run in one process.
