import numpy as np
import pandas as pd
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from mlex.utils.preprocessing import PreProcessingTransformer


class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, target_column, categories=None, filter_dict=None, **kwargs):
        """
        Initialize MLP model.
        
        Args:
            target_column: str - name of target column in dataset
            categories: list - categorical column values for preprocessing
            filter_dict: optional dictionary to filter data on (optional)
            **kwargs: additional model parameters
        """
        super().__init__()
        self.model_params = {
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
            'validation_fraction': kwargs.get('validation_fraction', None),
            'early_stopping': kwargs.get('early_stopping', None),
            'verbose': kwargs.get('verbose', None),
        }
        self.preprocessor_params = {
            'numeric_features': kwargs.get('numeric_features', None),
            'categorical_features': kwargs.get('categorical_features', None),
            'passthrough_features': kwargs.get('passthrough_features', None),
        }
        self.target_column = target_column
        self.categories = categories
        self.filter_dict = filter_dict
        self.final_model = None
        self.model = None

        self.fitted_ = False
        self.last_fit_time = 0

    @property
    def name(self):
        return 'MLP'

    def fit(self, X, y):
        if self.filter_dict is not None:
            X, y = self._apply_filter(X, y)

        self._set_categories(X)
        self.model = self._build_model()
        start = time.perf_counter()
        self.model.fit(X, y)
        end = time.perf_counter()

        self.last_fit_time = end - start
        self.fitted_ = True
        return self

    def predict(self, X):
        self._validate_fitted()
        
        if self.filter_dict is None:
            return self.model.predict(X)

        X_filtered = self._apply_filter(X)

        if len(X_filtered) == 0:
            return np.zeros(len(X))

        y_pred_filtered = self.model.predict(X_filtered)
        
        y_pred = np.zeros(len(X), dtype=y_pred_filtered.dtype)
        original_indices_of_filtered = X.index.get_indexer(X_filtered.index)
        y_pred[original_indices_of_filtered] = y_pred_filtered
        
        return y_pred

    def predict_proba(self, X):
        self._validate_fitted()
        
        if self.filter_dict is None:
            return self.model.predict_proba(X)[:, -1]

        X_filtered = self._apply_filter(X)

        if len(X_filtered) == 0:
            return np.zeros(len(X))

        y_pred_filtered = self.model.predict_proba(X_filtered)[:, -1]

        y_pred = np.zeros(len(X))
        original_indices_of_filtered = X.index.get_indexer(X_filtered.index)
        y_pred[original_indices_of_filtered] = y_pred_filtered
        
        return y_pred

    def _apply_filter(self, X, y=None):
        if self.filter_dict is None:
            return X, y

        mask = pd.Series(True, index=X.index)
        for col, val in self.filter_dict.items():
            if col in X.columns:
                if isinstance(val, list):
                    mask &= X[col].isin(val)
                else:
                    mask &= (X[col] == val)

        X_filtered = X[mask].copy()

        if y is not None:
            y_filtered = y[mask].copy()
            return X_filtered, y_filtered

        return X_filtered

    def _build_model(self):
        model_params = {
            'hidden_layer_sizes': self.model_params.get('hidden_layer_sizes', (10,)) or (10,),
            'activation': self.model_params.get('activation', 'relu') or 'relu',
            'solver': self.model_params.get('solver', 'adam') or 'adam',
            'batch_size': self.model_params.get('batch_size', 32) or 32,
            'shuffle': self.model_params.get('shuffle', True) if self.model_params.get('shuffle') is not None else True,
            'learning_rate': self.model_params.get('learning_rate', 'constant') or 'constant',
            'learning_rate_init': self.model_params.get('learning_rate_init', 1e-3) or 1e-3,
            'alpha': self.model_params.get('alpha', 0.0001) or 0.0001,
            'epsilon': self.model_params.get('epsilon', 1e-8) or 1e-8,
            'max_iter': self.model_params.get('max_iter', 100) or 100,
            'random_state': self.model_params.get('random_state', None) or None,
            'validation_fraction': self.model_params.get('validation_fraction', 0.3) or 0.3,
            'early_stopping': self.model_params.get('early_stopping', True) if self.model_params.get('early_stopping') is not None else True,
            'verbose': self.model_params.get('verbose', True) if self.model_params.get('verbose') is not None else True,
        }
        preprocessor_params = {
            'numeric_features': self.preprocessor_params.get('numeric_features', None) or None,
            'categorical_features': self.preprocessor_params.get('categorical_features', None) or None,
            'passthrough_features': self.preprocessor_params.get('passthrough_features', None) or None,
            'context_feature': self.preprocessor_params.get('context_feature', None) or None,
        }
        self.model_params.update(model_params)
        self.preprocessor_params.update(preprocessor_params)

        self.final_model = MLPClassifier(**{k: v for k, v in model_params.items()})
        preprocessor = PreProcessingTransformer(target_column=[self.target_column], **{k: v for k, v in preprocessor_params.items()}, categories=self.categories, handle_unknown='ignore')
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('final_model', self.final_model)
        ])

        return model

    def _set_categories(self, X):
        categorical_features = self.preprocessor_params.get('categorical_features', None)
        if self.categories is None and categorical_features is not None:
            self.categories = [X[col].unique() for col in categorical_features]

    def _validate_fitted(self):
        if not self.fitted_:
            raise ValueError("Model is not fitted")

    def get_feature_names(self):
        return self.model.named_steps['preprocessor'].get_feature_names_out()

    def get_params(self, deep=True):
        params = {**self.model_params, **self.preprocessor_params}.copy()
        params['target_column'] = self.target_column
        params['categories'] = self.categories
        return params

    def set_params(self, **parameters):
        if 'target_column' in parameters:
            self.target_column = parameters.pop('target_column')
        if 'categories' in parameters:
            self.categories = parameters.pop('categories')
        self.model_params.update({key: parameters[key] for key in list(self.model_params.keys()) if key in parameters})
        self.preprocessor_params.update({key: parameters[key] for key in list(self.preprocessor_params.keys()) if key in parameters})
        return self
