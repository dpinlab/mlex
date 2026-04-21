import time
from typing import Type

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from mlex.models.base_components.classical.params import (
    ClassicalPreprocessorParams,
    ClassicalWrapperConfigParams,
)
from mlex.utils.preprocessing import PreProcessingTransformer


class ClassicalModel(BaseEstimator, ClassifierMixin):
    ESTIMATOR_CLS: Type = None
    PARAMS_CLS: Type[BaseModel] = None
    NAME: str = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.ESTIMATOR_CLS is None:
            raise TypeError(f"{cls.__name__} must define ESTIMATOR_CLS")
        if cls.PARAMS_CLS is None:
            raise TypeError(f"{cls.__name__} must define PARAMS_CLS")
        if cls.NAME is None:
            raise TypeError(f"{cls.__name__} must define NAME")
        if not issubclass(cls.PARAMS_CLS, BaseModel):
            raise TypeError(
                f"{cls.__name__}.PARAMS_CLS must subclass BaseModel"
            )

    def __init__(self, **kwargs):
        super().__init__()
        self.config, self.model_params, self.preprocessor_params = self._split_kwargs(kwargs)

        self.final_model = None
        self.model = None
        self.fitted_ = False
        self.last_fit_time = 0

    @property
    def name(self):
        return self.NAME

    @property
    def target_column(self): return self.config.target_column
    @property
    def categories(self): return self.config.categories
    @categories.setter
    def categories(self, value): self.config = self.config.model_copy(update={'categories': value})
    @property
    def filter_dict(self): return self.config.filter_dict

    def _schemas(self):
        return (
            (ClassicalWrapperConfigParams, 'config'),
            (self.PARAMS_CLS, 'model_params'),
            (ClassicalPreprocessorParams, 'preprocessor_params'),
        )

    def _split_kwargs(self, kwargs):
        leftover = dict(kwargs)
        buckets = []
        for schema, _attr in self._schemas():
            bucket = {k: leftover.pop(k) for k in list(leftover) if k in schema.model_fields}
            buckets.append(schema(**bucket))
        if leftover:
            raise TypeError(
                f"unknown parameter(s): {sorted(leftover)}. "
                f"valid keys are declared in "
                f"{', '.join(s.__name__ for s, _ in self._schemas())}."
            )
        return tuple(buckets)

    def _update_params(self, updates):
        for schema, attr in self._schemas():
            bucket = {k: v for k, v in updates.items() if k in schema.model_fields}
            if bucket:
                setattr(self, attr, getattr(self, attr).model_copy(update=bucket))

    def fit(self, X, y):
        if self.filter_dict is not None:
            X, y = self._apply_filter(X, y)

        self._set_categories(X)
        self.model = self._build_model()
        start = time.perf_counter()
        self.model.fit(X, y)
        self.last_fit_time = time.perf_counter() - start
        self.fitted_ = True
        return self

    def predict(self, X):
        return self._predict_with_filter(X, proba=False)

    def predict_proba(self, X):
        return self._predict_with_filter(X, proba=True)

    def _predict_with_filter(self, X, proba):
        self._validate_fitted()

        def _call(x):
            return self.model.predict_proba(x)[:, -1] if proba else self.model.predict(x)

        if self.filter_dict is None:
            return _call(X)

        X_filtered = self._apply_filter(X)
        if len(X_filtered) == 0:
            return np.zeros(len(X))

        y_pred_filtered = _call(X_filtered)
        y_pred = np.zeros(len(X), dtype=y_pred_filtered.dtype)
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

    def _set_categories(self, X):
        categorical_features = self.preprocessor_params.categorical_features
        if self.categories is None and categorical_features is not None:
            self.categories = [X[col].unique() for col in categorical_features]

    def _validate_fitted(self):
        if not self.fitted_:
            raise ValueError("Model is not fitted")

    def _build_model(self):
        self.final_model = self.ESTIMATOR_CLS(**self.model_params.model_dump())
        preprocessor = PreProcessingTransformer(
            target_column=[self.target_column],
            **self.preprocessor_params.model_dump(),
            categories=self.categories,
            handle_unknown='ignore',
        )
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('final_model', self.final_model),
        ])

    def get_feature_names(self):
        return self.model.named_steps['preprocessor'].get_feature_names_out()

    def get_params(self, deep=True):
        return {
            **self.config.model_dump(),
            **self.model_params.model_dump(),
            **self.preprocessor_params.model_dump(),
        }

    def set_params(self, **parameters):
        self._update_params(parameters)
        return self
