# TD3 KL / KLE Negative Results Summary

## Experiment 1: Fixed-Beta TD3-KL

Result folder: `sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703`

Fixed-temperature KL soft-robust TD3 critic using the exponential-moment reparameterization

\[
f(s,a)=\exp\left(-Q(s,a)/(\gamma\beta)\right).
\]

The intended fixed-\(\beta\) backup is

\[
Q_\beta(s,a)
=
r(s,a)
-
\gamma\beta
\log
\mathbb{E}_{s'\sim P_0(\cdot|s,a)}
\left[
\exp\left(-V_\beta(s')/\beta\right)
\right].
\]

This first run used one replay next state for the moment target. It underperformed vanilla TD3 broadly because the KL variants lost nominal skill. On the focused Walker axes, vanilla nominal median return was about `1790`, while representative KL variants were much lower: `klb2` about `853`, `klb20` about `767`, and `klb100` about `605`.

Interpretation: the method produced conservative low-skill policies rather than robust high-skill policies. With only one nominal replay next state, the KL transition-support interpretation is also weak in deterministic MuJoCo dynamics.

## Experiment 2: KLE Next-Observation Ensemble Diagnostic

Result folder: `sweeps/results/TD3_KLE_Walker_diag_20260708_td3_kle_nextobs_diag`

Small empirical next-observation ensemble:

\[
-\beta \log \frac{1}{K}\sum_{k=1}^K \exp(-V(s'_k)/\beta),
\]

with \(K=8\), where the first \(s'_k\) is the replay next observation and the remaining samples are local noisy next-observation perturbations.

This did not rescue performance. In the focused Walker diagnostic, vanilla nominal median return was about `2343-2410`, while KLE variants were much lower: `kle100` about `675`, `kle2` about `506`, and `kle20` about `285-320` depending on the nominal axis. KLE100 slightly beat vanilla at exactly stress factor `0.5` for action replacement and actuator gain, but all policies were in a catastrophic low-return regime there. For mass, vanilla remained clearly better.

Interpretation: local observation perturbations give the KL moment non-degenerate support, but that support does not match the deployment perturbation geometry. Mass and actuator perturbations are physical transition shifts, not isotropic observation shifts.

## Takeaway

Simple TD3-KL critic-target modifications did not reproduce the useful robustness signal observed in PPO value-target clipping. Maybe KL transition robustness requires meaningful transition support. A more literal KL method would need domain-randomized replay, a learned dynamics ensemble, or one-step resimulation under physically structured perturbations.

Next test is TD3-TV value-space target clipping:

\[
y
=
r
+
\gamma(1-d)
\min\left(c,\min_i Q_{\bar\phi_i}(s',\tilde a')\right).
\]
