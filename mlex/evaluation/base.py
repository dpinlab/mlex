from abc import ABC, abstractmethod

import numpy as np


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, y_true, y_pred, scores):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def summary(self):
        pass

    @staticmethod
    def _is_binary(y):
        return len(np.unique(y)) == 2
