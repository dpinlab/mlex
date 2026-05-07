import time

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from mlex.evaluation.threshold import F1MaxThresholdStrategy
from mlex.models.base_components.recurrent._base import _RecurrentBaseModel
from mlex.models.base_components.recurrent.params import (
    PreprocessorParams,
    RecurrentModelParams,
    WrapperConfigParams,
)
from mlex.observers.observers import AUCROCObserver
from mlex.utils.context_aware import ContextAware
from mlex.utils.preprocessing import PreProcessingTransformer
from mlex.utils.split import FeatureStratifiedSplit, PastFutureSplit


_PARAM_SCHEMAS = (WrapperConfigParams, RecurrentModelParams, PreprocessorParams)


class RecurrentModel(nn.Module, BaseEstimator, ClassifierMixin):
    BASE_MODEL_CLS: type = None
    NAME: str = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.BASE_MODEL_CLS is None:
            raise TypeError(f"{cls.__name__} must define BASE_MODEL_CLS")
        if cls.NAME is None:
            raise TypeError(f"{cls.__name__} must define NAME")
        if not issubclass(cls.BASE_MODEL_CLS, _RecurrentBaseModel):
            raise TypeError(
                f"{cls.__name__}.BASE_MODEL_CLS must subclass _RecurrentBaseModel"
            )

    def __init__(self, **kwargs):
        super().__init__()
        self.config, self.model_params, self.preprocessor_params = self._split_kwargs(kwargs)

        self.train_data = None
        self.fitted_ = False
        self.final_model = None
        self.model = None
        self.threshold = 0.5
        self.last_fit_time = 0

        if self.model_params.input_size is not None:
            self.model = self._build_model()

    @property
    def name(self):
        return self.NAME

    # --- WrapperConfigParams fields exposed as attrs (read: delegate, write: rebuild schema) ---
    @property
    def target_column(self): return self.config.target_column
    @property
    def timestamp_column(self): return self.config.timestamp_column
    @property
    def val_data(self): return self.config.val_data
    @val_data.setter
    def val_data(self, value): self.config = self.config.model_copy(update={'val_data': value})
    @property
    def categories(self): return self.config.categories
    @categories.setter
    def categories(self, value): self.config = self.config.model_copy(update={'categories': value})
    @property
    def context_column(self): return self.config.context_column
    @property
    def val_split(self): return self.config.val_split
    @property
    def split_stratify_column(self): return self.config.split_stratify_column
    @property
    def sort_columns(self): return self.config.sort_columns
    @property
    def filter_dict(self): return self.config.filter_dict

    @staticmethod
    def _split_kwargs(kwargs):
        buckets = []
        leftover = dict(kwargs)
        for schema in _PARAM_SCHEMAS:
            schema_keys = schema.model_fields.keys()
            bucket = {k: leftover.pop(k) for k in list(leftover) if k in schema_keys}
            buckets.append(schema(**bucket))
        if leftover:
            raise TypeError(
                f"unknown parameter(s): {sorted(leftover)}. "
                f"valid keys are declared in "
                f"{', '.join(s.__name__ for s in _PARAM_SCHEMAS)}."
            )
        return tuple(buckets)

    def _update_params(self, updates):
        for schema, attr in (
            (WrapperConfigParams, 'config'),
            (RecurrentModelParams, 'model_params'),
            (PreprocessorParams, 'preprocessor_params'),
        ):
            bucket = {k: v for k, v in updates.items() if k in schema.model_fields}
            if bucket:
                setattr(self, attr, getattr(self, attr).model_copy(update=bucket))

    def fit(self, X, y, **kwargs):
        test_data = kwargs.pop('test_data', None)
        self._update_params(kwargs)

        X, y = self._prepare_input(X, y)
        self._set_categories(X)
        self.train_data = (X, y)
        self._prepare_validation_split(X, y)
        self._autoconfigure(test_data)

        start = time.perf_counter()
        self.model.fit(self.train_data[0], self.train_data[1])
        self.last_fit_time = time.perf_counter() - start

        self._set_threshold()
        self.fitted_ = True
        self.final_model.fitted_ = True
        return self

    def _autoconfigure(self, test_data=None):
        if self.model_params.input_size is not None:
            return

        preprocessor = PreProcessingTransformer(
            target_column=[self.target_column],
            **self.preprocessor_params.model_dump(),
            categories=self.categories,
            handle_unknown='ignore',
        )
        preprocessor.fit(self.train_data[0])
        feature_names = preprocessor.get_feature_names_out()
        X_val_transformed, y_val_transformed = preprocessor.transform(
            self.val_data[0], self.val_data[1]
        )

        observers = list(self.model_params.epoch_observers or [])
        if test_data is not None:
            X_test_transformed, y_test_transformed = preprocessor.transform(
                test_data[0], test_data[1]
            )
            observers.append(
                AUCROCObserver(name='test', data=(X_test_transformed, y_test_transformed))
            )

        self.model_params = self.model_params.model_copy(update={
            'feature_names': feature_names,
            'input_size': feature_names.shape[0] - 1,
            'validation_data': (X_val_transformed, y_val_transformed),
            'epoch_observers': observers,
        })
        self.model = self._build_model()

    def predict_proba(self, X):
        self._validate_fitted()
        X_sorted, _ = self._prepare_input(X)
        y_pred_sorted, seq_end_indices = self._predict_sorted(X_sorted)
        if len(seq_end_indices) == 0:
            return np.zeros(len(X))
        if self.model_params.collect_activations:
            if '__orig_index' in X_sorted.columns:
                self.activation_indices_ = X_sorted['__orig_index'].values[seq_end_indices]
            else:
                self.activation_indices_ = X_sorted.index[seq_end_indices]
        return self._map_to_original_order(X, X_sorted, y_pred_sorted)

    def predict(self, X):
        self._validate_fitted()
        y_pred = self.predict_proba(X)
        return (y_pred >= self.threshold).astype(int)

    def _prepare_input(self, X, y=None):
        X, y = self._apply_filter(X, y)
        return self._apply_context_aware_transform(X, y)

    def _apply_context_aware_transform(self, X, y=None):
        if self.timestamp_column is not None:
            context_sorter = ContextAware(
                target_column=self.target_column,
                timestamp_column=self.timestamp_column,
                context_column=self.context_column,
                sort_columns=self.sort_columns,
            )
            X, y = context_sorter.transform(X, y)
        return X, y

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

        return X_filtered, None

    def _set_categories(self, X):
        categorical_features = self.preprocessor_params.categorical_features
        if self.categories is None and categorical_features is not None:
            self.categories = [X[col].unique() for col in categorical_features]

    def _prepare_validation_split(self, X, y):
        if self.val_data is not None:
            return

        if self.split_stratify_column is not None:
            splitter = FeatureStratifiedSplit(
                stratify_column=self.split_stratify_column,
                split_proportion=self.val_split,
            )
        else:
            splitter = PastFutureSplit(
                proportion=self.val_split,
                timestamp_column=self.timestamp_column,
            )
        splitter.fit(X, y)
        X_train, y_train, X_val, y_val = splitter.transform(X, y)
        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)

    def _validate_fitted(self):
        if not self.fitted_:
            raise ValueError("Model is not fitted")

    def _assemble_sorted_predictions(self, X_sorted, end_positions, seq_scores):
        y_pred = np.zeros(len(X_sorted), dtype=float)
        y_pred[np.array(end_positions, dtype=int)] = seq_scores
        return y_pred

    def _map_to_original_order(self, X_original, X_sorted, y_pred_sorted):
        orig_indices = X_sorted['__orig_index'].values
        index_to_position = {idx: pos for pos, idx in enumerate(X_original.index)}

        y_pred = np.zeros(len(X_original), dtype=float)
        for sorted_pos, orig_idx in enumerate(orig_indices):
            if orig_idx in index_to_position:
                y_pred[index_to_position[orig_idx]] = y_pred_sorted[sorted_pos]

        return y_pred

    def _predict_sorted(self, X_sorted):
        seq_pred = self.model.predict(X_sorted)
        seq_end_indices = self.final_model.predict_end_indices
        y_pred_sorted = self._assemble_sorted_predictions(X_sorted, seq_end_indices, seq_pred)
        return y_pred_sorted, seq_end_indices

    def _set_threshold(self):
        y_pred, _ = self._predict_sorted(self.val_data[0])
        threshold_strat = F1MaxThresholdStrategy()
        self.threshold = threshold_strat.compute_threshold(self.val_data[1], y_pred)

    def _build_model(self):
        preprocessor = PreProcessingTransformer(
            target_column=[self.target_column],
            **self.preprocessor_params.model_dump(),
            categories=self.categories,
            handle_unknown='ignore',
        )
        self.final_model = self.BASE_MODEL_CLS(**self.model_params.model_dump())
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('final_model', self.final_model),
        ])

    def get_feature_names(self):
        return self.model_params.feature_names

    def get_params(self, deep=True):
        return {
            **self.config.model_dump(),
            **self.model_params.model_dump(),
            **self.preprocessor_params.model_dump(),
            'threshold': self.threshold,
        }

    def set_params(self, **parameters):
        if 'threshold' in parameters:
            self.threshold = parameters.pop('threshold')
        self._update_params(parameters)
        return self

    def get_y_true_sequences(self, X, y):
        self._validate_fitted()
        return y.values.flatten()
