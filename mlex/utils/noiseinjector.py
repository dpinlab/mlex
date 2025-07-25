from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import random


class BaseNoiseStrategy:
    def inject_noise(self, y, noise_percentage, target_columns):
        raise NotImplementedError


class ProportionalNoiseStrategy(BaseNoiseStrategy):
    def inject_noise(self, y, noise_percentage, target_columns):
        if y is None:
            return y
        y_noisy = y.copy()
        y_values = y_noisy[target_columns].values if hasattr(y_noisy, 'values') else y_noisy
        y_noisy_values = y_values.copy()
        if y_noisy_values.ndim > 1 and y_noisy_values.shape[1] == 1:
            y_noisy_values = y_noisy_values.ravel()
        noise_perc = noise_percentage / 100.0
        idx_0 = np.where(y_noisy_values == 0)[0]
        idx_1 = np.where(y_noisy_values == 1)[0]
        n_0 = len(idx_0)
        n_1 = len(idx_1)
        n_flip_1_to_0 = int(round(n_1 * noise_perc))
        n_flip_0_to_1 = min(n_flip_1_to_0, n_0)
        flip_idx_1_to_0 = np.random.choice(idx_1, size=n_flip_1_to_0, replace=False) if n_flip_1_to_0 > 0 and n_1 > 0 else []
        flip_idx_0_to_1 = np.random.choice(idx_0, size=n_flip_0_to_1, replace=False) if n_flip_0_to_1 > 0 and n_0 > 0 else []
        y_noisy_values[flip_idx_1_to_0] = 0
        y_noisy_values[flip_idx_0_to_1] = 1
        if hasattr(y_noisy, 'values'):
            y_noisy[target_columns] = y_noisy_values.reshape(-1, 1)
        else:
            y_noisy = y_noisy_values

        return y_noisy


class RandomNoiseStrategy(BaseNoiseStrategy):
    def inject_noise(self, y, noise_percentage, target_columns):
        if y is None:
            return  y
        y_noisy = y.copy()
        y_values = y_noisy[target_columns].values if hasattr(y_noisy, 'values') else y_noisy
        y_noisy_values = y_values.copy()

        noise_lenght = len(y_noisy_values) * (noise_percentage/100)
        chosed_transactions_indexs = random.sample(range(0, len(y_noisy_values)), k=int(noise_lenght))

        y_noisy_values[chosed_transactions_indexs] = 1 - y_noisy_values[chosed_transactions_indexs]

        if hasattr(y_noisy, 'values'):
            y_noisy[target_columns] = y_noisy_values.reshape(-1, 1)
        else:
            y_noisy = y_noisy_values
        return y_noisy



class NoiseInjector(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: BaseNoiseStrategy, noise_percentage=10, target_columns=None):
        self.strategy = strategy
        self.noise_percentage = noise_percentage
        self.target_columns = target_columns or ['I-d']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is None:
            return X, y
        y_noisy = self.strategy.inject_noise(y, self.noise_percentage, self.target_columns)
        return X, y_noisy
