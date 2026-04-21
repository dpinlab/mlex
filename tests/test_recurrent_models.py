import unittest

import numpy as np
import pydantic
import torch
import torch.nn as nn
from parameterized import parameterized

from mlex.models import BILSTM, GRU, LSTM, RNN
from mlex.models.base_components.recurrent import (
    BILSTMBaseModel,
    GRUBaseModel,
    LSTMBaseModel,
    PreprocessorParams,
    RNNBaseModel,
    RecurrentModel,
    RecurrentModelParams,
    _RecurrentBaseModel,
)


WRAPPERS = [("RNN", RNN), ("LSTM", LSTM), ("BILSTM", BILSTM), ("GRU", GRU)]
BASE_MODELS = [
    ("RNN", RNNBaseModel, nn.RNN, False),
    ("LSTM", LSTMBaseModel, nn.LSTM, False),
    ("BILSTM", BILSTMBaseModel, nn.LSTM, True),
    ("GRU", GRUBaseModel, nn.GRU, False),
]


def _make_data(n_samples=60, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples).astype(np.float32).reshape(-1, 1)
    return X, y


class RecurrentModelParamsTests(unittest.TestCase):
    def test_defaults_match_legacy_behavior(self):
        p = RecurrentModelParams()
        self.assertEqual(p.hidden_size, 10)
        self.assertEqual(p.num_layers, 1)
        self.assertEqual(p.seq_length, 30)
        self.assertEqual(p.epochs, 30)
        self.assertEqual(p.learning_rate, 1e-3)
        self.assertTrue(p.shuffle_dataloader)
        self.assertIsNone(p.input_size)
        self.assertIsNone(p.epoch_observers)

    def test_positive_int_fields_reject_non_positive(self):
        for field in ("hidden_size", "num_layers", "seq_length", "batch_size", "epochs"):
            with self.subTest(field=field):
                with self.assertRaises(pydantic.ValidationError):
                    RecurrentModelParams(**{field: 0})

    def test_extra_field_forbidden(self):
        with self.assertRaises(pydantic.ValidationError):
            RecurrentModelParams(unknown_hyperparam=42)

    def test_preprocessor_params_default_context_feature_is_isolated(self):
        a = PreprocessorParams()
        b = PreprocessorParams()
        a.context_feature.append("LEAK")
        self.assertEqual(b.context_feature, ["CONTEXT"])


class SubclassGuardTests(unittest.TestCase):
    def test_base_model_subclass_without_layer_cls_raises(self):
        with self.assertRaises(TypeError):
            class BadBase(_RecurrentBaseModel):
                pass

    def test_wrapper_subclass_without_base_model_cls_raises(self):
        with self.assertRaises(TypeError):
            class BadWrapper(RecurrentModel):
                NAME = "bad"

    def test_wrapper_subclass_without_name_raises(self):
        with self.assertRaises(TypeError):
            class BadWrapper(RecurrentModel):
                BASE_MODEL_CLS = RNNBaseModel

    def test_wrapper_rejects_non_recurrent_base_model(self):
        with self.assertRaises(TypeError):
            class BadWrapper(RecurrentModel):
                BASE_MODEL_CLS = nn.Linear
                NAME = "bad"


class BaseModelArchitectureTests(unittest.TestCase):
    @parameterized.expand(BASE_MODELS)
    def test_recurrent_layer_type_and_linear_shape(self, _name, base_cls, layer_cls, bidirectional):
        X_val, y_val = _make_data(n_samples=40, seed=1)
        model = base_cls(
            validation_data=(X_val, y_val),
            input_size=3,
            hidden_size=5,
            epochs=1,
            seq_length=4,
            batch_size=8,
            group_index=None,
            device=torch.device("cpu"),
        )
        self.assertIsInstance(model.recurrent_layer, layer_cls)
        expected_linear_in = 5 * (2 if bidirectional else 1)
        self.assertEqual(model.linear.in_features, expected_linear_in)
        if bidirectional:
            self.assertTrue(model.recurrent_layer.bidirectional)

    @parameterized.expand(BASE_MODELS)
    def test_fit_runs_and_records_history(self, _name, base_cls, _layer_cls, _bidi):
        X_val, y_val = _make_data(n_samples=40, seed=2)
        X_train, y_train = _make_data(n_samples=80, seed=3)
        model = base_cls(
            validation_data=(X_val, y_val),
            input_size=3,
            hidden_size=4,
            epochs=2,
            patience=5,
            seq_length=4,
            batch_size=8,
            group_index=None,
            random_seed=0,
            device=torch.device("cpu"),
        )
        model.fit(X_train, y_train)
        self.assertEqual(len(model.history["epoch"]), 2)
        self.assertEqual(len(model.history["train"]), 2)
        self.assertEqual(len(model.history["val"]), 2)


class WrapperIdentityTests(unittest.TestCase):
    @parameterized.expand(WRAPPERS)
    def test_name_matches_class(self, name, cls):
        instance = cls(target_column="y", timestamp_column="t")
        self.assertEqual(instance.name, name)

    @parameterized.expand(WRAPPERS)
    def test_params_round_trip_through_get_set(self, _name, cls):
        m = cls(
            target_column="y",
            timestamp_column="t",
            hidden_size=64,
            epochs=7,
            numeric_features=["a", "b"],
        )
        m2 = cls(target_column="y", timestamp_column="t").set_params(**m.get_params())
        self.assertEqual(m.get_params(), m2.get_params())
        self.assertEqual(m2.model_params.hidden_size, 64)
        self.assertEqual(m2.model_params.epochs, 7)
        self.assertEqual(m2.preprocessor_params.numeric_features, ["a", "b"])

    @parameterized.expand(WRAPPERS)
    def test_construction_validates_hyperparams(self, _name, cls):
        with self.assertRaises(pydantic.ValidationError):
            cls(target_column="y", timestamp_column="t", hidden_size=-1)

    @parameterized.expand(WRAPPERS)
    def test_missing_required_params_raise(self, _name, cls):
        with self.assertRaises(pydantic.ValidationError):
            cls()
        with self.assertRaises(pydantic.ValidationError):
            cls(target_column="y")
        with self.assertRaises(pydantic.ValidationError):
            cls(timestamp_column="t")

    @parameterized.expand(WRAPPERS)
    def test_unknown_kwarg_raises(self, _name, cls):
        with self.assertRaises(TypeError) as ctx:
            cls(target_column="y", timestamp_column="t", totally_made_up=42)
        self.assertIn("totally_made_up", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
