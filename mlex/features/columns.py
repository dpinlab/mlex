import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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


class AutoMappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_arr = np.array(X, dtype=object)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        transformed_cols = [
            pd.factorize(X_arr[:, i], sort=False)[0] 
            for i in range(X_arr.shape[1])
        ]
        
        return np.column_stack(transformed_cols)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return None


class CompositeTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, numeric_features, categorical_features, automap_features, 
                 categories='auto', handle_unknown='error') -> None:
        super().__init__()
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.automap_features = automap_features
        self.encoder =  ColumnTransformer(   
            transformers=[
                ("num", NumericalTransfomer(), self.numeric_features),
                ("cat", CategoricalOneHotTransfomer(categories=categories,handle_unknown=handle_unknown), self.categorical_features),
                ("auto_mapped", AutoMappingTransformer(), self.automap_features),
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