# Catastrophic robustness analysis

The reliability curve uses a fixed threshold relative to the median nominal
return of vanilla TD3. Its AUC equals the expected normalized return clipped
to [0, 1], so unusually large lucky returns cannot hide catastrophic failures.

Metrics are first computed within each perturbation axis and then averaged
equally across axes. This prevents a family with a denser factor grid from
receiving more weight.

The variance decomposition separates:
- persistent seed quality: seed main-effect variance;
- perturbation severity: perturbation main-effect variance;
- seed-specific fragility: seed-by-perturbation interaction variance.

The final term measures whether different training seeds fail under different
deployment shifts, beyond a seed being uniformly good or bad.

`perturbation_recovery_regimes.csv` uses an explicitly labeled ex-post
axis oracle over models that retain at least 80% nominal return. It measures
whether a failure family is recoverable by the trained menu; it is not a
zero-target-budget deployment claim. Fixed-model aggregate rows provide the
corresponding zero-budget comparison.

`model_delta_seed_bootstrap.csv` resamples training seeds independently for
each method and vanilla. It therefore does not assume that equal numeric seed
labels create meaningful policy pairs. It also reports the common-language
probability that a randomly selected robust seed improves over a randomly
selected vanilla seed.
