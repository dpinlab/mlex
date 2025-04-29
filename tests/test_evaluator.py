import unittest

import numpy as np

from mlex.evaluation.evaluator import StandardEvaluator
from mlex.evaluation.threshold import F1MaxThresholdStrategy


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 0, 1, 1])
        self.scores = np.array([0.1, 0.3, 0.6, 0.8])
        self.evaluator = StandardEvaluator("test", F1MaxThresholdStrategy())

    def test_evaluation(self):
        self.evaluator.evaluate(self.y_true, np.array([0, 0, 1, 1]), self.scores)
        metrics = self.evaluator.results['metrics']
        self.assertTrue(metrics['accuracy'] >= 0.75)

    def test_save_load(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            self.evaluator.evaluate(self.y_true, np.array([0, 0, 1, 1]), self.scores)
            self.evaluator.save(file_path)

            self.assertTrue(os.path.exists(file_path))
            self.assertGreater(os.path.getsize(file_path), 0)

            self.evaluator.load(file_path)
            self.assertIsNotNone(self.evaluator.results)


if __name__ == '__main__':
    unittest.main()
