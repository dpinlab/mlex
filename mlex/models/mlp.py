import torch.nn as nn
import numpy as np
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from mlex.utils.preprocessing import PreProcessingTransformer


class MLP(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(self, target_column=None, categories=None, **kwargs):
        """
        Initialize MLP model.
        
        Args:
            target_column: str - name of target column in dataset
            categories: list - categorical column values for preprocessing
            **kwargs: additional model parameters
        """
        super().__init__()
        self.params = {
            'hidden_layer_sizes': kwargs.get('hidden_layer_sizes', None),
            'activation': kwargs.get('activation', None),
            'solver': kwargs.get('solver', None),
            'batch_size': kwargs.get('batch_size', None),
            'shuffle': kwargs.get('shuffle', None),
            'learning_rate': kwargs.get('learning_rate', None),
            'learning_rate_init': kwargs.get('learning_rate_init', None),
            'alpha': kwargs.get('alpha', None),
            'epsilon': kwargs.get('epsilon', None),
            'max_iter': kwargs.get('max_iter', None),
            'random_state': kwargs.get('random_state', None),
            'feature_names': kwargs.get('feature_names', None),
            'validation_fraction': kwargs.get('validation_fraction', None),
            'early_stopping': kwargs.get('early_stopping', None),
            'verbose': kwargs.get('verbose', None),
            'numeric_features': kwargs.get('numeric_features', None),
            'categorical_features': kwargs.get('categorical_features', None),
            'passthrough_features': kwargs.get('passthrough_features', None),
            'automap_features': kwargs.get('automap_features', None),
        }
        self.target_column = target_column
        self.categories = categories
        self.final_model = None
        self.model = None

        self.model = self._build_model()

        self.last_fit_time = 0

    @property
    def name(self):
        return 'MLP'

    def fit(self, X, y):
        preprocessor = PreProcessingTransformer(target_columns=[self.target_column], **{k: v for k, v in self.params.items() if '_features' in k}, categories=self.categories, handle_unknown='ignore')
        self.params['feature_names'] = preprocessor.get_feature_names_out()

        start = time.perf_counter()
        self.model.fit(X, y)
        end = time.perf_counter()

        self.last_fit_time = end - start
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, -1]

    def score_samples(self, X):
        return self.model.score_samples(X)

    def _build_model(self):
        model_params = {
            'hidden_layer_sizes': self.params.get('hidden_layer_sizes', (10,)) or (10,),
            'activation': self.params.get('activation', 'relu') or 'relu',
            'solver': self.params.get('solver', 'adam') or 'adam',
            'batch_size': self.params.get('batch_size', 32) or 32,
            'shuffle': self.params.get('shuffle', True) if self.params.get('shuffle') is not None else True,
            'learning_rate': self.params.get('learning_rate', 'constant') or 'constant',
            'learning_rate_init': self.params.get('learning_rate_init', 1e-3) or 1e-3,
            'alpha': self.params.get('alpha', 0.0001) or 0.0001,
            'epsilon': self.params.get('epsilon', 1e-8) or 1e-8,
            'max_iter': self.params.get('max_iter', 100) or 100,
            'random_state': self.params.get('random_state', None) or None,
            'validation_fraction': self.params.get('validation_fraction', 0.3) or 0.3,
            'early_stopping': self.params.get('early_stopping', True) if self.params.get('early_stopping') is not None else True,
            'verbose': self.params.get('verbose', True) if self.params.get('verbose') is not None else True,
            'numeric_features': self.params.get('numeric_features', ['DIA_LANCAMENTO','MES_LANCAMENTO','VALOR_TRANSACAO','VALOR_SALDO']),
            'categorical_features': self.params.get('categorical_features', ['TIPO', 'CNAB', 'NATUREZA_SALDO']),
            'passthrough_features': self.params.get('passthrough_features', None),
            'automap_features': self.params.get('automap_features', None),
        }
        self.params.update(model_params)

        self.final_model = MLPClassifier(**{k: v for k, v in model_params.items() if not '_features' in k})
        preprocessor = PreProcessingTransformer(target_columns=[self.target_column], **{k: v for k, v in model_params.items() if '_features' in k}, categories=self.categories, handle_unknown='ignore')
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('final_model', self.final_model)
        ])

        return model

    def get_feature_names(self):
        return self.params.get('feature_names')

    def get_params(self, deep=True):
        return self.params.copy()

    def set_params(self, **parameters):
        self.params.update(parameters)
        return self
