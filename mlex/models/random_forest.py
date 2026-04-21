import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlex.models.base_components.classical import ClassicalModel, RandomForestParams


class RandomForest(ClassicalModel):
    ESTIMATOR_CLS = RandomForestClassifier
    PARAMS_CLS = RandomForestParams
    NAME = 'RandomForest'

    def feature_importances(self):
        self._validate_fitted()
        importances = self.model.named_steps['final_model'].feature_importances_
        return pd.Series(importances, index=self.get_feature_names()).sort_values(ascending=False)

    def permutation_importances(self, X, y, n_repeats=10, random_state=None):
        self._validate_fitted()
        from sklearn.inspection import permutation_importance
        X = self.model.named_steps['preprocessor'].transform(X)
        r = permutation_importance(
            self.model.named_steps['final_model'],
            X,
            y.values.flatten(),
            n_repeats=n_repeats,
            random_state=random_state,
        )
        return pd.Series(r.importances_mean, index=self.get_feature_names()).sort_values(ascending=False)

    def decision_path(self, X):
        self._validate_fitted()
        X = self.model.named_steps['preprocessor'].transform(X)
        return self.model.named_steps['final_model'].decision_path(X)
