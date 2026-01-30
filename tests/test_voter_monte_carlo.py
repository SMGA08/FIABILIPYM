import unittest

import numpy as np

from fiabilipym import Component, Voter


class DeterministicComponent(Component):
    def sample_failure_time(self, rng, size=1):
        if size == 3:
            return np.array([1.0, 5.0, 9.0])
        return np.full(size, 5.0, dtype=float)


class TestVoterMonteCarlo(unittest.TestCase):
    def test_time_to_failure_sample(self):
        component = DeterministicComponent("C", 1.0)
        voter = Voter(component, M=2, N=3)
        rng = np.random.default_rng(0)
        self.assertEqual(voter.time_to_failure_sample(rng), 5.0)


if __name__ == "__main__":
    unittest.main()
