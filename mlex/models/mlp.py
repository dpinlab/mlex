from sklearn.neural_network import MLPClassifier

from mlex.models.base_components.classical import ClassicalModel, MLPParams


class MLP(ClassicalModel):
    ESTIMATOR_CLS = MLPClassifier
    PARAMS_CLS = MLPParams
    NAME = 'MLP'
