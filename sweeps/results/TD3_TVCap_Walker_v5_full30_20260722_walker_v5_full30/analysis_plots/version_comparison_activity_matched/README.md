# TD3 TV-Cap Cross-Version Comparison

Reference: Walker2d-v4, model `tvc250`

Candidate: Walker2d-v5, model `tvc400`

All returns are normalized by the corresponding environment version's
cross-seed vanilla nominal median. Perturbation axes receive equal weight.
Bootstrap samples treat each independently trained policy seed as one cluster.
The cross-version interaction is the candidate-version robust-minus-vanilla
effect minus the reference-version robust-minus-vanilla effect. It therefore
does not confuse a raw reward-scale change with replication of the clipping
effect.
