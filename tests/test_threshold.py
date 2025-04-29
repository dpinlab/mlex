import unittest

import numpy as np

from mlex.evaluation.threshold import QuantileThresholdStrategy, F1MaxThresholdStrategy


class TestThresholdStrategies(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    def test_quantile_strategy(self):
        strategy = QuantileThresholdStrategy(quantile=95)
        threshold = strategy.compute_threshold(self.y_true, self.scores)
        self.assertAlmostEqual(threshold, 0.95, places=2)

    def test_f1_max_strategy(self):
        strategy = F1MaxThresholdStrategy()
        threshold = strategy.compute_threshold(self.y_true, self.scores)
        self.assertTrue(0.5 <= threshold <= 0.7)


if __name__ == '__main__':
    unittest.main()
