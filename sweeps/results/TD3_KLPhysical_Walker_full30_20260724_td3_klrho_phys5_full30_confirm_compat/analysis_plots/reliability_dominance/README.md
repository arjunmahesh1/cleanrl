# Reliability dominance analysis

For normalized return \(X=R/R_0\), define
\[
S_m(t)=P(X_m\geq t),\qquad 0\leq t\leq 1.
\]
The difference \(S_m(t)-S_0(t)\) is the catastrophe-probability reduction at
threshold \(t\). Positive values favor the robust model.

The signed area equals the clipped normalized-return improvement:
\[
\int_0^1 [S_m(t)-S_0(t)]\,dt
=
\mathbb E[\operatorname{clip}(X_m,0,1)]
-
\mathbb E[\operatorname{clip}(X_0,0,1)].
\]
`catastrophe_recovery_area` integrates the positive part, while
`catastrophe_harm_area` integrates the negative part. A zero harm area and
100% dominance-threshold fraction mean the empirical robust reliability curve
never falls below vanilla on the evaluated threshold grid. This is a
descriptive empirical dominance statement, not a population proof.

Curves are computed within each perturbation axis and then averaged equally
over axes so grid density does not determine the result.

Uncertainty is estimated by independently resampling robust and vanilla
training seeds while keeping every perturbation level from a sampled seed
together. The shaded bands are pointwise 95% intervals. The reported
`probability_curve_empirically_dominates` is the fraction of bootstrap
replicates whose minimum reliability difference over the full threshold grid
is nonnegative; it is stricter than inspecting pointwise intervals. The
normalizing \(R_0\) is the full-sample vanilla nominal median, so these
intervals are conditional on the benchmark's chosen empirical reference.
