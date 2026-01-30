import unittest

import numpy as np

from fiabilipym import Component, System, Weibull, weibull_eta_from_lambda


def build_series(components):
    system = System()
    system["E"] = components[0]
    system[components[0]] = components[1]
    system[components[1]] = components[2]
    system[components[2]] = "S"
    return system


def build_parallel(components):
    system = System()
    system["E"] = components
    for comp in components:
        system[comp] = "S"
    return system


def build_series_parallel(components):
    system = System()
    system["E"] = components[0]
    system[components[0]] = [components[1], components[2]]
    system[components[1]] = "S"
    system[components[2]] = "S"
    return system


class TestMonteCarloSystems(unittest.TestCase):
    def setUp(self):
        beta = 1.0
        lam = 1e-4
        eta = weibull_eta_from_lambda(lam, beta)
        dist = Weibull(beta, eta)

        self.components = [
            Component("C0", lam).with_distribution(dist),
            Component("C1", lam).with_distribution(dist),
            Component("C2", lam).with_distribution(dist),
        ]

    def test_mttf_ordering(self):
        series = build_series(self.components)
        series_parallel = build_series_parallel(self.components)
        parallel = build_parallel(self.components)

        grid_t = np.linspace(0, 50000, 5)
        mttf_series, _ = series.monte_carlo(n=6000, grid_t=grid_t, seed=1)
        mttf_series_parallel, _ = series_parallel.monte_carlo(
            n=6000, grid_t=grid_t, seed=2
        )
        mttf_parallel, _ = parallel.monte_carlo(n=6000, grid_t=grid_t, seed=3)

        self.assertLess(mttf_series, mttf_series_parallel)
        self.assertLess(mttf_series_parallel, mttf_parallel)

    def test_monte_carlo_convergence(self):
        series = build_series(self.components)
        grid_t = np.linspace(0, 50000, 3)
        mttf_small, _ = series.monte_carlo(n=2000, grid_t=grid_t, seed=7)
        mttf_large, _ = series.monte_carlo(n=8000, grid_t=grid_t, seed=7)

        rel_diff = abs(mttf_large - mttf_small) / mttf_large
        self.assertLess(rel_diff, 0.3)


class TestWeibullHelpers(unittest.TestCase):
    def test_weibull_mean_match(self):
        beta = 2.0
        lam = 5e-4
        eta = weibull_eta_from_lambda(lam, beta)
        dist = Weibull(beta, eta)
        self.assertAlmostEqual(dist.mean(), 1.0 / lam, places=6)


if __name__ == "__main__":
    unittest.main()
