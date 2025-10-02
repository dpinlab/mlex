import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ..features.columns import CompositeTransformer
import random


class PreProcessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column=None, numeric_features=None, categorical_features=None, passthrough_features=None, context_feature=None, categories=None, handle_unknown='error'):
        self.target_column = target_column or []
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.passthrough_features = passthrough_features or []
        self.context_feature = context_feature or []

        self.composite = CompositeTransformer(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            passthrough_features=self.passthrough_features,
            context_feature=self.context_feature,
            categories=categories,
            handle_unknown=handle_unknown
        )
        self.feature_names_ = None


    def fit(self, X, y=None):
        feature_cols = self.numeric_features + self.categorical_features + self.passthrough_features + self.context_feature
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
        feature_cols = self.numeric_features + self.categorical_features + self.passthrough_features + self.context_feature
        X_transformed = self.composite.transform(X[feature_cols])

        y_out = None
        if y is not None:
            y_out = np.nan_to_num(y[self.target_column].values)
            return X_transformed, y_out

        return X_transformed


    def get_feature_names_out(self, input_features=None):
        return self.feature_names_
