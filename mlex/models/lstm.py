from mlex.models.base_components.recurrent import LSTMBaseModel, RecurrentModel


class LSTM(RecurrentModel):
    BASE_MODEL_CLS = LSTMBaseModel
    NAME = 'LSTM'
