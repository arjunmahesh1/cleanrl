# PPO 30-Seed Confirmatory Robustness Results

This folder contains targeted 30-seed confirmatory evaluations for PPO value-target clipping.

## Purpose

The goal is to test whether robustness effects observed in the 5-seed full perturbation grid persist under a larger number of training seeds.

These are targeted confirmatory experiments, not full-grid evaluations.

## Walker2d

Training horizon: 1,000,000 environment steps.

Models:
- vanilla PPO
- TV cap = 2.95
- TV cap = 3.10

Perturbations:
- `mass_0p5`: body mass scale 0.5
- `actuator_gain_0p5`: actuator gain scale 0.5

Files:
- `Walker2d/combined_metrics.csv`: all individual seed/model/scenario eval rows
- `Walker2d/summary_by_model.csv`: mean, std, median, SEM, and 95% CI by model/scenario
- `Walker2d/paired_gain_over_vanilla.csv`: paired seed-wise gain of each cap over vanilla

## HalfCheetah

Training horizon: 5,000,000 environment steps.

Models:
- vanilla PPO
- TV cap = 2.55
- TV cap = 2.65
- TV cap = 2.75

Perturbations:
- `actuator_gain_0p5`: actuator gain scale 0.5
- `mass_damping_0p5`: body mass scale 0.5 and joint damping scale 0.5

Files:
- `HalfCheetah/combined_metrics.csv`: all individual seed/model/scenario eval rows
- `HalfCheetah/summary_by_model.csv`: mean, std, median, SEM, and 95% CI by model/scenario
- `HalfCheetah/paired_gain_over_vanilla.csv`: paired seed-wise gain of each cap over vanilla

## Interpretation

The key confirmatory columns are:
- `mean_gain_over_vanilla`
- `ci95_gain`
- `fraction_seeds_beating_vanilla`

A robust effect is more convincing when the paired gain is positive, the confidence interval is mostly positive, and the cap beats vanilla on a large fraction of seeds.
