# PPO Normalized Alpha Calibration (2026-03-25)

Goal: calibrate fixed-alpha clipping for the **normalized PPO pipeline** using the
late-training distribution of PPO value targets, not episodic returns.

Procedure:
- Train `vanilla` PPO for 5 seeds with the standard normalized pipeline.
- Use late-training values of:
  - `targets/returns_p95_pre_transform`
  - `targets/returns_p99_pre_transform`
  - `targets/returns_max_pre_transform`
- Aggregate across seeds with the median.
- Build the coarse alpha grid around that target scale.

Why:
- In this implementation, clipping acts on PPO value targets (`b_returns`), not
  on episodic return.
- Therefore alpha must be sized in target space.

Current manually copied calibration notes:
- Median `p95`: `2.9275`
- Median `p99`: `2.9623`
- Median `max`: `2.9977`

Recommended coarse normalized grid:
- `vanilla`
- `noop` (`a1e9` control)
- `2.85`
- `2.95`
- `3.05`
- `3.20`
- `3.50`

What to verify in the coarse grid:
- `robust/tv_return_clip_fraction` turns on late, not immediately.
- `robust/tv_return_excess_mean` is nonzero for active alphas.
- Nominal `charts/episodic_return` does not collapse versus vanilla.
- `noop` tracks `vanilla`.

Files in this folder:
- `seed_summary.csv`: manually copied late-run calibration values from W&B.
- Add cluster logs / copied run metadata here after extraction.
