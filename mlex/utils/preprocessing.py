import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..features.columns import CompositeTransformer
import random


class PreProcessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_columns=None, numeric_features=None, categorical_features=None, categories='auto', handle_unknown='error'):
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
        self.composite = CompositeTransformer(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            categories=categories,
            handle_unknown=handle_unknown
        )
        self.feature_names_ = None
        self.y_ = None

    def fit(self, X, y=None):
        feature_cols = self.numeric_features + self.categorical_features
        self.composite.fit(X[feature_cols])
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

    def transform(self, X, y, noise_percentage=10, insert_noise=False):
        feature_cols = self.numeric_features + self.categorical_features
        X_transformed = self.composite.transform(X[feature_cols])

        self.feature_names_ = self.composite.get_feature_names_out()
        if y is not None:
            self.y_ = np.nan_to_num(y[self.target_columns].values)

            if insert_noise == True:
                self.y_ = self.inserting_noise(self.y_, noise_percentage)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

    def get_target(self):
        return self.y_
