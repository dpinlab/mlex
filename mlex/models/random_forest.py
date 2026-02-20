import numpy as np
import pandas as pd
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from mlex.utils.preprocessing import PreProcessingTransformer


class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, target_column, categories=None, filter_dict=None, **kwargs):
        """
        Initialize RandomForest model.
        
        Args:
            target_column: str - name of target column in dataset
            categories: list - categorical column values for preprocessing
            filter_dict: optional dictionary to filter data on (optional)
            **kwargs: additional model parameters
        """
        super().__init__()
        self.model_params = {
            'n_estimators': kwargs.get('n_estimators', None),
            'criterion': kwargs.get('criterion', None),
            'max_depth': kwargs.get('max_depth', None),
            'min_samples_split': kwargs.get('min_samples_split', None),
            'min_samples_leaf': kwargs.get('min_samples_leaf', None),
            'min_weight_fraction_leaf': kwargs.get('min_weight_fraction_leaf', None),
            'max_features': kwargs.get('max_features', None),
            'max_leaf_nodes': kwargs.get('max_leaf_nodes', None),
            'min_impurity_decrease': kwargs.get('min_impurity_decrease', None),
            'bootstrap': kwargs.get('bootstrap', None),
            'oob_score': kwargs.get('oob_score', None),
            'n_jobs': kwargs.get('n_jobs', None),
            'random_state': kwargs.get('random_state', None),
            'verbose': kwargs.get('verbose', None),
            'warm_start': kwargs.get('warm_start', None),
            'class_weight': kwargs.get('class_weight', None),
            'ccp_alpha': kwargs.get('ccp_alpha', None),
            'max_samples': kwargs.get('max_samples', None),
            'monotonic_cst': kwargs.get('monotonic_cst', None),
        }
        self.preprocessor_params = {
            'numeric_features': kwargs.get('numeric_features', None),
            'categorical_features': kwargs.get('categorical_features', None),
            'passthrough_features': kwargs.get('passthrough_features', None),
            'context_feature': kwargs.get('context_feature', None),
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
        return 'RandomForest'

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

        y_pred = np.zeros(len(X))
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
            'n_estimators': self.model_params.get('n_estimators', 100) or 100,
            'criterion': self.model_params.get('criterion', 'gini') or 'gini',
            'max_depth': self.model_params.get('max_depth', None) or None,
            'min_samples_split': self.model_params.get('min_samples_split', 2) or 2,
            'min_samples_leaf': self.model_params.get('min_samples_leaf', 1) or 1,
            'min_weight_fraction_leaf': self.model_params.get('min_weight_fraction_leaf', 0.0) or 0.0,
            'max_features': self.model_params.get('max_features', 'sqrt') or 'sqrt',
            'max_leaf_nodes': self.model_params.get('max_leaf_nodes', None) or None,
            'min_impurity_decrease': self.model_params.get('min_impurity_decrease', 0.0) or 0.0,
            'bootstrap': self.model_params.get('bootstrap', True) if self.model_params.get('bootstrap') is not None else True,
            'oob_score': self.model_params.get('oob_score', False) if self.model_params.get('oob_score') is not None else False,
            'n_jobs': self.model_params.get('n_jobs', None) or None,
            'random_state': self.model_params.get('random_state', None) or None,
            'verbose': self.model_params.get('verbose', True) if self.model_params.get('verbose') is not None else True,
            'warm_start': self.model_params.get('warm_start', False) if self.model_params.get('warm_start') is not None else False,
            'class_weight': self.model_params.get('class_weight', None) or None,
            'ccp_alpha': self.model_params.get('ccp_alpha', 0.0) or 0.0,
            'max_samples': self.model_params.get('max_samples', None) or None,
            'monotonic_cst': self.model_params.get('monotonic_cst', None) or None,
        }
        preprocessor_params = {
            'numeric_features': self.preprocessor_params.get('numeric_features', None) or None,
            'categorical_features': self.preprocessor_params.get('categorical_features', None) or None,
            'passthrough_features': self.preprocessor_params.get('passthrough_features', None) or None,
            'context_feature': self.preprocessor_params.get('context_feature', None) or None,
        }
        self.model_params.update(model_params)
        self.preprocessor_params.update(preprocessor_params)

        self.final_model = RandomForestClassifier(**{k: v for k, v in model_params.items()})
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

    def feature_importances(self):
        self._validate_fitted()
        importances = self.model.named_steps['final_model'].feature_importances_
        return pd.Series(importances, index=self.get_feature_names()).sort_values(ascending=False)

    def permutation_importances(self, X, y, n_repeats=10, random_state=None):
        self._validate_fitted()
        from sklearn.inspection import permutation_importance
        X = self.model.named_steps['preprocessor'].transform(X)
        r = permutation_importance(self.model.named_steps['final_model'], X, y.values.flatten(), n_repeats=n_repeats, random_state=random_state)
        return pd.Series(r.importances_mean, index=self.get_feature_names()).sort_values(ascending=False)

    def decision_path(self, X):
        self._validate_fitted()
        X = self.model.named_steps['preprocessor'].transform(X)
        return self.model.named_steps['final_model'].decision_path(X)

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
