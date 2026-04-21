from mlex.models.base_components.recurrent import RecurrentModel, RNNBaseModel


class RNN(RecurrentModel):
    BASE_MODEL_CLS = RNNBaseModel
    NAME = 'RNN'
