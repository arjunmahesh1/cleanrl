# Random Perturbation Experiments (Legacy TV90 Stage)

This folder documents earlier random-perturbation experiments that preceded the
current friction-only 5-seed run.

## Source datasets

- `sweeps/walker2d_adversarial_stage1.csv` (100 perturbation combinations)
- `sweeps/smoke_adversarial_select.csv` (small selected subset used for quick checks)
- `sweeps/smoke.csv` (early smoke results)

## Notes

- These are mostly single-seed style results and should be treated as exploratory.
- Metric columns use the older naming (`tv90_*`) from the earlier variant.
- They are useful as evidence that some perturbations can strongly change returns,
  and can be used to seed later adversarial-search ranges.

## Highlighted rows from `smoke_adversarial_select.csv`

| mass | friction | damping | TV nominal mean | TV perturbed mean | TV delta | Vanilla nominal mean | Vanilla perturbed mean | Vanilla delta | Gain (vanilla drop - TV drop) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | 0.9 | 0.8 | 2181.88 | 1546.82 | -635.06 | 3855.31 | 4661.85 | +806.54 | -1441.60 |
| 1.1 | 0.9 | 1.2 | 2181.88 | 5332.22 | +3150.34 | 3855.31 | 6188.46 | +2333.15 | +817.18 |

Interpretation of the positive row (`mass=1.1, friction=0.9, damping=1.2`):
- Both models improved under this perturbation.
- TV improved more than vanilla in that single-seed check (`+817` gain).
- This is encouraging but not a robust claim without multi-seed confirmation.
