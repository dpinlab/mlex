import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
import itertools as ite

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class CategoricalOneHotTransfomer(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder = OneHotEncoder()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.encoder.fit_transform(X).toarray()
        return Xt

    def get_feature_names_out(self, input_features=None):
        return self.encoder.get_feature_names_out(input_features)
    

class NumericalTransfomer(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder = MinMaxScaler()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.encoder.fit_transform(X)
        return Xt

    def get_feature_names_out(self, input_features=None):
        return input_features
    
class CompositeTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, numeric_features, categorical_features) -> None:
        super().__init__()
        self.numberic_feature = numeric_features
        self.categorical_features = categorical_features
        self.encoder =  ColumnTransformer(   
            transformers=[
                ("num", NumericalTransfomer(), self.numberic_feature),
                ("cat", CategoricalOneHotTransfomer(), self.categorical_features),
            ],
            verbose_feature_names_out=False
        )

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.encoder.fit_transform(X, y)
        return Xt

    def get_feature_names_out(self, input_features=None):
        return self.encoder.get_feature_names_out(input_features)

#TODO implementar
class EmbeedinglTransfomer(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
        # self.encoder = Embeding(handle_unknown="ignore")
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        # Xt = self.encoder.fit_transform(X)
        return X