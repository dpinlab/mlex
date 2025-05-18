import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..features.columns import CompositeTransformer


class PreProcessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_columns=None, numeric_features=None, categorical_features=None):
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
            categorical_features=self.categorical_features
        )
        self.feature_names_ = None
        self.y_ = None

    def fit(self, X, y=None):
        feature_cols = self.numeric_features + self.categorical_features
        self.composite.fit(X[feature_cols])
        return self

    def transform(self, X, y):
        feature_cols = self.numeric_features + self.categorical_features
        X_transformed = self.composite.transform(X[feature_cols])

        self.feature_names_ = self.composite.get_feature_names_out()
        self.y_ = np.nan_to_num(y[self.target_columns].values)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

    def get_target(self):
        return self.y_
