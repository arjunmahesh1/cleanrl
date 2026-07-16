# Walker2d TD3 TV-Cap: Paper Interpretation

## Experiment

- 30 independently trained seeds per model.
- Eight models: vanilla TD3 and caps `c = 100, 150, 200, 225, 250, 275, 300`.
- One million environment steps per policy.
- 20 perturbation axes and 20 evaluation episodes per row.
- 71,520 evaluation rows, or 1,430,400 evaluated episodes.

The TD3 target is

```text
q_next = min(Q1_target(s', a'), Q2_target(s', a'))
y_c = r + gamma * (1 - done) * min(c, q_next)
```

The cap is therefore a value-scale pessimism parameter. It is not itself a TV radius. Smaller `c` means a stronger truncation of the upper tail of the bootstrapped target.

## Main Result

TD3 TV-capping is not uniformly beneficial. The full experiment shows three cap-activity regimes:

1. `c = 100, 150`: strongly active and destructive. Median run-average ClipFraction is 0.359 and 0.114, respectively. These caps suppress the learned Q scale and substantially reduce nominal and perturbed returns.
2. `c = 200, 225`: transition regime. Clipping is seed-dependent and performance recovers, but the median robustness score remains below vanilla.
3. `c = 250, 275, 300`: mostly inactive. Only 13/30, 9/30, and 1/30 seeds ever exceed 1% ClipFraction. These variants approach vanilla, so apparent gains at `c = 300` should not be presented as evidence for a strong clipping mechanism.

The strongest defensible local result is the coupled friction+mass+damping perturbation at factor 0.5. For `c = 250` relative to same-seed vanilla:

- mean return gain: +226.5;
- median return gain: +137.9;
- wins: 21/30 seeds;
- 95% paired confidence interval: [23.6, 429.3];
- paired t-test: p = 0.030;
- Wilcoxon signed-rank test: p = 0.026;
- unpaired Welch test: p = 0.035;
- Mann-Whitney test: p = 0.056;
- reliability AUC: 0.811 versus 0.709 for vanilla.

The same cap shows smaller, statistically uncertain gains at factor 0.5 for mass+damping (+101.4 mean; 17/30 wins), mass (+86.5; 16/30), and friction+mass. This is evidence of a recoverable coupled-physics regime, not a universal improvement.

Action replacement is qualitatively different. At probability 0.5, `c = 100` improves mean return by 70.2 and wins 25/30 paired seeds, but every model remains below the catastrophe threshold of 50% of vanilla nominal return. This is a relative improvement inside an irrecoverable regime, not restored robustness.

## Aggregate Result

Using the same nominal-preserving robustness score as the PPO analysis,

```text
Score(c, p) = min(nominal retention, median non-nominal retention),
```

vanilla wins 12 of 20 axes, `c = 250` wins six, and `c = 300` wins two. The median score is 0.900 for vanilla, 0.821 for `c = 250`, and 0.844 for `c = 300`. Thus there is no zero-budget cap that dominates vanilla over the complete perturbation suite.

At representative stress factor 0.5, `c = 300` has the highest median reliability AUC (0.854 versus 0.837 for vanilla), but it clips meaningfully in only one seed. The safest interpretation is that a loose, nearly inactive cap preserves nominal TD3 behavior; it does not establish that active TV pessimism is broadly superior.

## Seed Variance

The cap menu is most useful when vanilla TD3 trains poorly. Across six selected adversarial axes, the ex-post best cap has Pearson correlation -0.377 with vanilla nominal return and produces positive gain in 21/30 seeds. Mean ex-post gain is +252 for weak seeds, +262 for middle seeds, and only +19 for elite seeds.

This is a model-selection statement, not a deployable zero-budget guarantee. For the fixed preselected cap `c = 250`, mean gain is +135 for weak seeds, +43 for middle seeds, and -218 for elite seeds; its overall mean gain is slightly negative.

## Novel Diagnostic

A fixed numerical cap does not impose a fixed effective robustness budget. For `c = 250`, 13 seeds ever clip meaningfully while 17 are essentially inactive. Active seeds have larger absolute gains, but they also begin from higher-Q, higher-return policies. In fact, greater ClipFraction is negatively associated with improvement in return retention after nominal performance is factored out.

ClipFraction is therefore endogenous to the learned value scale. The next theoretically cleaner experiment is to target a fixed ClipFraction `rho` during training by adapting `c_t`, then compare policies at matched effective pessimism. This converts the current observation into a falsifiable question: does robustness improve because of clipping itself, or because the cap selects particular training trajectories?

## Validity Note

All 240 policies saved successfully, and the critic, actor, TD-target, Q-scale, ClipFraction, and excess-target diagnostics are present. The TensorBoard `charts/episodic_return` and episode-length tags are missing because the training logger did not recognize Gymnasium's `final_info` format. This is a logging defect, not an environment reset, replay, target, optimization, model-saving, or evaluation defect. Policy quality comparisons should use the complete post-training deterministic evaluation table, not the incomplete training-event endpoint field.
