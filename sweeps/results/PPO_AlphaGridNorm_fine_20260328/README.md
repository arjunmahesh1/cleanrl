# PPO Alpha Grid - Normalized Fine Sweep

**Date:** 2026-03-28
**WandB group:** `ppo-alpha-norm-fine-expanded`
**Project:** `fixed-alpha-randomness`

## Extraction procedure

- Runs are pulled from W&B group `ppo-alpha-norm-fine-expanded`.
- Variant labels are derived from the training config (`vanilla`, `noop`, or the fixed cap value).
- This report uses **final run.summary values**.
- IQM means: sort the 5 seed-level values, drop the min and max, average the middle 3.

## High-level readout

- `vanilla` nominal IQM return: **3628.0**.
- `noop` nominal IQM return: **3628.0**.
- Current best active variant by the scripted rule: **`3.05`**.

## Per-variant summary

| Variant | Nominal IQM return | IQM clip fraction | IQM excess mean | Eval IQM return |
|---|---:|---:|---:|---:|
| vanilla | 3628.0 | - | - | 2565.0 |
| noop | 3628.0 | 0.0 | 0.0 | 2565.0 |
| 2.95 | 2944.0 | 0.0127 | 0.0003 | 1726.0 |
| 3.00 | 3150.0 | 0.0 | 0.0 | 2862.0 |
| 3.05 | 3238.0 | 0.0112 | 0.0001 | 3072.0 |
| 3.10 | 3281.0 | 0.0 | 0.0 | 1842.0 |
| 3.50 | 2378.0 | 0.0 | 0.0 | 2331.0 |
| 3.70 | 3144.0 | 0.0 | 0.0 | 2726.0 |
| 4.00 | 2584.0 | 0.0 | 0.0 | 3631.0 |

## Files

```text
outputs/
  alpha_grid_fine_summary.txt
  alpha_grid_fine_summary.csv
  alpha_grid_fine_summary.png
```

## Notes

- Use the W&B training curves for detailed timing of when clipping turns on or fades out.
- Use the last checkpoint policy for final robustness evaluation.

