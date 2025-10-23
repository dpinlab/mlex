import torch.nn as nn
import numpy as np
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from mlex.models.base_components.lstm_base_model import LSTMBaseModel
from mlex.utils.preprocessing import PreProcessingTransformer


class LSTM(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(self, validation_data, target_column=None, categories=None, **kwargs):
        """
        Initialize LSTM model.
        
        Args:
            validation_data: tuple of (X_val, y_val) - validation features and targets
            target_column: str - name of target column in dataset
            categories: list - categorical column values for preprocessing
            **kwargs: additional model parameters
        """
        super().__init__()
        self.model_params = {
            'input_size': kwargs.get('input_size', None),
            'hidden_size': kwargs.get('hidden_size', None),
            'num_layers': kwargs.get('num_layers', None),
            'output_size': kwargs.get('output_size', None),
            'seq_length': kwargs.get('seq_length', None),
            'batch_size': kwargs.get('batch_size', None),
            'shuffle_dataloader': kwargs.get('shuffle_dataloader', None),
            'learning_rate': kwargs.get('learning_rate', None),
            'alpha': kwargs.get('alpha', None),
            'eps': kwargs.get('eps', None),
            'weight_decay': kwargs.get('weight_decay', None),
            'epochs': kwargs.get('epochs', None),
            'patience': kwargs.get('patience', None),
            'group_index': kwargs.get('group_index', None),
            'random_seed': kwargs.get('random_seed', None),
            'feature_names': kwargs.get('feature_names', None),
            'device': kwargs.get('device', None),
            'validation_data': validation_data,  # tuple of (X_val, y_val)
        }
        self.preprocessor_params = {
            'numeric_features': kwargs.get('numeric_features', None) or None,
            'categorical_features': kwargs.get('categorical_features', None) or None,
            'passthrough_features': kwargs.get('passthrough_features', None) or None,
            'context_feature': kwargs.get('context_feature', None) or None,
        }
        self.target_column = target_column
        self.categories = categories
        self.final_model = None
        self.model = None

        if self.model_params['input_size'] is not None:
            self.model = self._build_model()

        self.last_fit_time = 0

    @property
    def name(self):
        return 'LSTM'

    def fit(self, X, y, **kwargs):
        # Update params with any new values
        self.model_params.update({key: kwargs[key] for key in list(self.model_params.keys()) if key in kwargs})
        self.preprocessor_params.update({key: kwargs[key] for key in list(self.preprocessor_params.keys()) if key in kwargs})

        if self.model_params['input_size'] is None:
            preprocessor = PreProcessingTransformer(target_column=[self.target_column], **{k: v for k, v in self.preprocessor_params.items()}, categories=self.categories, handle_unknown='ignore')
            preprocessor.fit(X)
            self.model_params['feature_names'] = preprocessor.get_feature_names_out()
            self.model_params['input_size'] = self.model_params['feature_names'].shape[0] - 1
            self.model_params['validation_data'] = preprocessor.transform(self.model_params['validation_data'][0], self.model_params['validation_data'][1])
            self.model = self._build_model()

        start = time.perf_counter()
        self.model.fit(X, y)
        end = time.perf_counter()
        
        self.last_fit_time = end - start
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        return self.model.score_samples(X)

    def _build_model(self):
        # Provide hardcoded defaults if still None
        model_params = {
            'input_size': self.model_params.get('input_size', 10) or 10,
            'hidden_size': self.model_params.get('hidden_size', 10) or 10,
            'num_layers': self.model_params.get('num_layers', 1) or 1,
            'output_size': self.model_params.get('output_size', 1) or 1,
            'seq_length': self.model_params.get('seq_length', 30) or 30,
            'batch_size': self.model_params.get('batch_size', 32) or 32,
            'shuffle_dataloader': self.model_params.get('shuffle_dataloader', True) if self.model_params.get('shuffle_dataloader') is not None else True,
            'learning_rate': self.model_params.get('learning_rate', 1e-3) or 1e-3,
            'alpha': self.model_params.get('alpha', .9) or .9,
            'eps': self.model_params.get('eps', 1e-7) or 1e-7,
            'weight_decay': self.model_params.get('weight_decay', 0.0) or 0.0,
            'epochs': self.model_params.get('epochs', 30) or 30,
            'patience': self.model_params.get('patience', 5) or 5,
            'group_index': self.model_params.get('group_index', -1) or -1,
            'random_seed': self.model_params.get('random_seed', None),
            'device': self.model_params.get('device', None),
            'validation_data': self.model_params.get('validation_data', None),
        }
        preprocessor_params = {
            'numeric_features': self.preprocessor_params.get('numeric_features', None) or None,
            'categorical_features': self.preprocessor_params.get('categorical_features', None) or None,
            'passthrough_features': self.preprocessor_params.get('passthrough_features', None) or None,
            'context_feature': self.preprocessor_params.get('context_feature', None) or None,
        }
        self.model_params.update(model_params)
        self.preprocessor_params.update(preprocessor_params)

        preprocessor = PreProcessingTransformer(target_column=[self.target_column], **{k: v for k, v in preprocessor_params.items()}, categories=self.categories, handle_unknown='ignore')
        self.final_model = LSTMBaseModel(validation_data=model_params['validation_data'], **{k: v for k, v in model_params.items() if k != 'validation_data'})
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('final_model', self.final_model)
        ])

        return model

    def get_feature_names(self):
        return self.model_params.get('feature_names')

    def get_params(self, deep=True):
        return {**self.model_params, **self.preprocessor_params}.copy()

    def set_params(self, **parameters):
        self.model_params.update({key: parameters[key] for key in list(self.model_params.keys()) if key in parameters})
        self.preprocessor_params.update({key: parameters[key] for key in list(self.preprocessor_params.keys()) if key in parameters})
        return self

    def create_test_loader(self, X, y):
        X = self.model.named_steps['preprocessor'].transform(X)
        return self.final_model._create_dataloader(X, y, shuffle_dataloader=False)
    
    def get_y_true_sequences(self, X, y):
        test_loader = self.create_test_loader(X, y)
        y_true = []
        for _, y_batch in test_loader:
            y_true.extend(np.array(y_batch, dtype="int8").flatten())
        return y_true
