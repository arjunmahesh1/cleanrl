import unittest

import torch

from cleanrl.td3_continuous_action import (
    kl_constrained_discrete_target,
    kl_regularized_discrete_target,
)


class TestTD3KLTargets(unittest.TestCase):
    def test_single_member_is_exact_td3_target(self):
        returns = torch.tensor([[1.25], [-3.0], [400.0]])
        log_weights = torch.zeros(1)
        regularized, weights = kl_regularized_discrete_target(
            returns,
            beta=0.01,
            log_reference_weights=log_weights,
        )
        constrained, constrained_weights, _, achieved = kl_constrained_discrete_target(
            returns,
            radius=1.0,
            log_reference_weights=log_weights,
            bisection_steps=40,
        )
        torch.testing.assert_close(regularized, returns[:, 0])
        torch.testing.assert_close(constrained, returns[:, 0])
        torch.testing.assert_close(weights, torch.ones_like(returns))
        torch.testing.assert_close(constrained_weights, torch.ones_like(returns))
        torch.testing.assert_close(achieved, torch.zeros(len(returns)))

    def test_zero_radius_is_reference_expectation(self):
        returns = torch.tensor([[1.0, 2.0, 5.0], [3.0, -2.0, 7.0]])
        weights = torch.tensor([0.6, 0.3, 0.1])
        target, adversary, beta, achieved = kl_constrained_discrete_target(
            returns,
            radius=0.0,
            log_reference_weights=torch.log(weights),
            bisection_steps=40,
        )
        torch.testing.assert_close(target, (returns * weights).sum(dim=1))
        torch.testing.assert_close(adversary, weights.expand_as(returns))
        self.assertTrue(torch.isinf(beta).all())
        torch.testing.assert_close(achieved, torch.zeros(len(returns)))

    def test_saturated_radius_uses_minimum_kl_worst_distribution(self):
        returns = torch.tensor([[1.0, 1.0, 3.0, 4.0, 5.0]])
        log_weights = torch.log(torch.full((5,), 0.2))
        target, adversary, beta, achieved = kl_constrained_discrete_target(
            returns,
            radius=1.0,
            log_reference_weights=log_weights,
            bisection_steps=60,
        )
        torch.testing.assert_close(target, torch.tensor([1.0]))
        torch.testing.assert_close(
            adversary,
            torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(beta, torch.zeros(1))
        torch.testing.assert_close(achieved, torch.tensor([torch.log(torch.tensor(2.5))]))

    def test_explicit_radius_is_scale_invariant(self):
        returns = torch.tensor(
            [[120.0, 160.0, 210.0, 300.0, 450.0], [30.0, 50.0, 55.0, 90.0, 140.0]]
        )
        log_weights = torch.log(torch.tensor([0.6, 0.1, 0.1, 0.1, 0.1]))
        target, adversary, beta, achieved = kl_constrained_discrete_target(
            returns,
            radius=0.1,
            log_reference_weights=log_weights,
            bisection_steps=60,
        )
        scaled_target, scaled_adversary, scaled_beta, scaled_achieved = (
            kl_constrained_discrete_target(
                0.01 * returns,
                radius=0.1,
                log_reference_weights=log_weights,
                bisection_steps=60,
            )
        )
        torch.testing.assert_close(scaled_target, 0.01 * target, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(scaled_adversary, adversary, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(scaled_beta, 0.01 * beta, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(scaled_achieved, achieved, rtol=2e-5, atol=2e-5)

    def test_explicit_radius_is_monotone_in_pessimism(self):
        returns = torch.tensor([[10.0, 20.0, 35.0, 50.0, 80.0]])
        log_weights = torch.log(torch.full((5,), 0.2))
        targets = []
        worst_weights = []
        for radius in (0.0, 0.05, 0.1, 0.2):
            target, adversary, _, achieved = kl_constrained_discrete_target(
                returns,
                radius=radius,
                log_reference_weights=log_weights,
                bisection_steps=60,
            )
            targets.append(float(target.item()))
            worst_weights.append(float(adversary[0, 0].item()))
            if radius > 0:
                self.assertAlmostEqual(float(achieved.item()), radius, places=5)
        self.assertTrue(all(left >= right for left, right in zip(targets, targets[1:])))
        self.assertTrue(
            all(left <= right for left, right in zip(worst_weights, worst_weights[1:]))
        )

    def test_targets_are_translation_equivariant(self):
        returns = torch.tensor([[1.0, 4.0, 9.0], [-5.0, 2.0, 8.0]])
        log_weights = torch.log(torch.tensor([0.5, 0.3, 0.2]))
        shift = 123.0
        target, adversary, _, achieved = kl_constrained_discrete_target(
            returns,
            radius=0.1,
            log_reference_weights=log_weights,
            bisection_steps=60,
        )
        shifted_target, shifted_adversary, _, shifted_achieved = (
            kl_constrained_discrete_target(
                returns + shift,
                radius=0.1,
                log_reference_weights=log_weights,
                bisection_steps=60,
            )
        )
        torch.testing.assert_close(shifted_target, target + shift)
        torch.testing.assert_close(shifted_adversary, adversary)
        torch.testing.assert_close(shifted_achieved, achieved)


if __name__ == "__main__":
    unittest.main()
