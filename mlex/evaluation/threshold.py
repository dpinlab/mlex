import numpy as np
from sklearn.metrics import precision_recall_curve


class ThresholdStrategy:
    def compute_threshold(self, y_true, scores):
        raise NotImplementedError


class QuantileThresholdStrategy(ThresholdStrategy):
    def __init__(self, quantile=.95):
        self.quantile = quantile

    def compute_threshold(self, y_true, scores):
        # if scores.ndim == 2 and scores.shape[1] == 2:
        #     scores = scores[:, 1]
        return np.quantile(scores, self.quantile)


class F1MaxThresholdStrategy(ThresholdStrategy):
    def compute_threshold(self, y_true, scores):
        # if scores.ndim == 2 and scores.shape[1] == 2:
        #     scores = scores[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        return thresholds[np.argmax(f1_scores[:-1])]
