import unittest

import numpy as np
import pandas as pd
import pydantic
from parameterized import parameterized
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from mlex.models import MLP, RandomForest
from mlex.models.base_components.classical import (
    ClassicalModel,
    ClassicalWrapperConfigParams,
    MLPParams,
    RandomForestParams,
)


WRAPPERS = [
    ("MLP", MLP, MLPClassifier, MLPParams),
    ("RandomForest", RandomForest, RandomForestClassifier, RandomForestParams),
]


def _make_df(n_samples=80, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'num1': rng.standard_normal(n_samples),
        'num2': rng.standard_normal(n_samples),
        'y': rng.integers(0, 2, size=n_samples),
    })


class ClassicalParamSchemaTests(unittest.TestCase):
    def test_mlp_defaults_match_legacy(self):
        p = MLPParams()
        self.assertEqual(p.hidden_layer_sizes, (10,))
        self.assertEqual(p.activation, 'relu')
        self.assertEqual(p.max_iter, 100)
        self.assertTrue(p.early_stopping)

    def test_rf_defaults_match_legacy(self):
        p = RandomForestParams()
        self.assertEqual(p.n_estimators, 100)
        self.assertEqual(p.criterion, 'gini')
        self.assertEqual(p.max_features, 'sqrt')
        self.assertTrue(p.bootstrap)

    def test_config_requires_target_column(self):
        with self.assertRaises(pydantic.ValidationError):
            ClassicalWrapperConfigParams()

    def test_extra_field_forbidden(self):
        with self.assertRaises(pydantic.ValidationError):
            MLPParams(unknown_hyperparam=1)
        with self.assertRaises(pydantic.ValidationError):
            RandomForestParams(unknown_hyperparam=1)


class SubclassGuardTests(unittest.TestCase):
    def test_missing_estimator_cls_raises(self):
        with self.assertRaises(TypeError):
            class Bad(ClassicalModel):
                PARAMS_CLS = MLPParams
                NAME = "bad"

    def test_missing_params_cls_raises(self):
        with self.assertRaises(TypeError):
            class Bad(ClassicalModel):
                ESTIMATOR_CLS = MLPClassifier
                NAME = "bad"

    def test_non_pydantic_params_cls_raises(self):
        with self.assertRaises(TypeError):
            class Bad(ClassicalModel):
                ESTIMATOR_CLS = MLPClassifier
                PARAMS_CLS = dict
                NAME = "bad"


class WrapperIdentityTests(unittest.TestCase):
    @parameterized.expand(WRAPPERS)
    def test_name_and_estimator(self, name, cls, estimator_cls, _params_cls):
        m = cls(target_column="y")
        self.assertEqual(m.name, name)
        self.assertIs(m.ESTIMATOR_CLS, estimator_cls)

    @parameterized.expand(WRAPPERS)
    def test_missing_target_column_raises(self, _name, cls, _est, _params):
        with self.assertRaises(pydantic.ValidationError):
            cls()

    @parameterized.expand(WRAPPERS)
    def test_unknown_kwarg_raises(self, _name, cls, _est, _params):
        with self.assertRaises(TypeError) as ctx:
            cls(target_column="y", totally_made_up=42)
        self.assertIn("totally_made_up", str(ctx.exception))

    @parameterized.expand(WRAPPERS)
    def test_params_round_trip(self, _name, cls, _est, _params):
        m = cls(target_column="y", numeric_features=["a"])
        m2 = cls(target_column="y").set_params(**m.get_params())
        self.assertEqual(m.get_params(), m2.get_params())


class MLPFitPredictTests(unittest.TestCase):
    def test_fit_predict_end_to_end(self):
        df = _make_df(seed=1)
        X, y = df[['num1', 'num2']], df[['y']]
        model = MLP(
            target_column='y',
            numeric_features=['num1', 'num2'],
            max_iter=20,
            hidden_layer_sizes=(4,),
            random_state=0,
            verbose=False,
            early_stopping=False,
        )
        model.fit(X, y)
        self.assertTrue(model.fitted_)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        self.assertEqual(preds.shape[0], len(X))
        self.assertEqual(probs.shape[0], len(X))


class RandomForestFitPredictTests(unittest.TestCase):
    def test_fit_predict_and_importances(self):
        df = _make_df(seed=2)
        X, y = df[['num1', 'num2']], df[['y']]
        model = RandomForest(
            target_column='y',
            numeric_features=['num1', 'num2'],
            n_estimators=5,
            random_state=0,
            verbose=False,
        )
        model.fit(X, y)
        self.assertEqual(model.predict(X).shape[0], len(X))
        self.assertEqual(model.predict_proba(X).shape[0], len(X))

        importances = model.feature_importances()
        self.assertEqual(set(importances.index), {'num1', 'num2'})
        self.assertEqual(len(importances), 2)


if __name__ == "__main__":
    unittest.main()
