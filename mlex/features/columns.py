import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
import itertools as ite

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class CategoricalOneHotTransfomer(BaseEstimator, TransformerMixin):
    
    def __init__(self, categories='auto', handle_unknown='error') -> None:
        super().__init__()
        self.categories = categories
        self.handle_unknown=handle_unknown
        self.encoder = OneHotEncoder(categories=categories, handle_unknown=handle_unknown)
        
    def fit(self, X, y=None):
        self.encoder.fit(X)
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
        self.encoder.fit(X)
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.encoder.transform(X)
        return Xt

    def get_feature_names_out(self, input_features=None):
        return input_features
    
class CompositeTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, numeric_features, categorical_features, categories='auto', handle_unknown='error') -> None:
        super().__init__()
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.encoder =  ColumnTransformer(   
            transformers=[
                ("num", NumericalTransfomer(), self.numeric_features),
                ("cat", CategoricalOneHotTransfomer(categories=categories,handle_unknown=handle_unknown), self.categorical_features),
            ],
            verbose_feature_names_out=False
        )
    def fit_transform(self, X, y = None, **fit_params):
        return self.encoder.fit_transform(X, y, **fit_params)
    
    def fit(self, X, y=None):
        self.encoder.fit(X,y)
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.encoder.transform(X)
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