# Training and deployment failure channels

Let `R0,p` be the vanilla cross-seed nominal median for axis `p`, and let
`t=0.5`. A seed-axis policy is nominally competent when its nominal
return is at least `t R0,p`.

The analysis reports:

- nominal training failure: `P(R_nominal < t R0,p)`;
- unconditional deployment failure under non-nominal perturbations;
- deployment failure conditional on nominal competence;
- reliability AUC conditional on nominal competence.

Axes receive equal weight. Bootstrap resampling treats one complete trained
policy seed, including all axes and perturbation levels, as the sampling
cluster. Robust and `vanilla` seeds are resampled
independently.

Conditional results are diagnostic rather than primary performance claims:
conditioning removes failed training runs and therefore changes the deployed
policy population. The unconditional metric remains the correct
algorithm-level result. The conditional metric asks whether an observed
failure came mainly from optimization reliability or from sensitivity after
a competent policy had been learned.
