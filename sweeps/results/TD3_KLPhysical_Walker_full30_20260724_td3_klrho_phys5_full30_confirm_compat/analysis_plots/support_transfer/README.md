# Physical-support transfer analysis

This analysis was fixed before the independent confirmation evaluation
completed. Axes are partitioned into:

- `direct_support`: global mass and global actuator gain, the two perturbed
  dynamics families present in the five-member training support;
- `related_physics`: localized actuator-gain or mass axes and global
  combinations involving mass; and
- `out_of_support`: all remaining deployment perturbations.

Within each axis, return is normalized by that axis's vanilla nominal median.
Reliability AUC is the mean clipped normalized return over non-nominal
conditions. Axes receive equal weight within each support class. Confidence
intervals independently bootstrap training seeds and retain every axis-level
summary from a sampled seed.

The support-alignment contrast is
\[
\bigl(\Delta\mathrm{AUC}_{\mathrm{direct}}\bigr)
-
\bigl(\Delta\mathrm{AUC}_{\mathrm{out}}\bigr).
\]
A positive contrast means the algorithm's relative benefit tracks the
geometry represented in its training support. It does not by itself show an
overall robustness improvement; the class-specific AUC deltas must also be
positive.

Robust model: `klprho0p05`.
Ensemble control: `klprho0`.
Baseline: `vanilla`.
