import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ..features.columns import CompositeTransformer
import random


class PreProcessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_columns=None, numeric_features=None, categorical_features=None, automap_features=None, categories=None, handle_unknown='error'):
        self.target_columns = target_columns or ['I-d']  # Default target
        self.numeric_features = numeric_features or [
            'DIA_LANCAMENTO',
            'MES_LANCAMENTO',
            'VALOR_TRANSACAO',
            'VALOR_SALDO',
        ]
        self.categorical_features = categorical_features or [
            'TIPO',
            'CNAB',
            'NATUREZA_SALDO'
        ]
        self.automap_features = automap_features or [
            'GROUP'
        ]

        self.composite = CompositeTransformer(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            automap_features=self.automap_features,
            categories=categories,
            handle_unknown=handle_unknown
        )
        self.feature_names_ = None


    def fit(self, X, y=None):
        feature_cols = self.numeric_features + self.categorical_features + self.automap_features
        self.composite.fit(X[feature_cols])
        self.feature_names_ = self.composite.get_feature_names_out()
        return self


    def inserting_noise(self, y, noise_percentage):
        noise_lenght_percentage = noise_percentage

        noise_lenght = len(y) * (noise_lenght_percentage/100)
        chosed_transactions_indexs = random.sample(range(0, len(y)), k=int(noise_lenght))

        for i in chosed_transactions_indexs:
            if y[i] == 0:
                y[i] = 1
            else:
                y[i] = 0
        return y


    def transform(self, X, y=None):
        feature_cols = self.numeric_features + self.categorical_features + self.automap_features
        X_transformed = self.composite.transform(X[feature_cols])

        y_out = None
        if y is not None:
            y_out = np.nan_to_num(y[self.target_columns].values)
            return X_transformed, y_out

        return X_transformed


    def get_feature_names_out(self, input_features=None):
        return self.feature_names_
