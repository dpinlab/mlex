import torch.nn as nn

from mlex.models.base_components.recurrent._base import _RecurrentBaseModel


class RNNBaseModel(_RecurrentBaseModel):
    LAYER_CLS = nn.RNN


class LSTMBaseModel(_RecurrentBaseModel):
    LAYER_CLS = nn.LSTM


class GRUBaseModel(_RecurrentBaseModel):
    LAYER_CLS = nn.GRU


class BILSTMBaseModel(_RecurrentBaseModel):
    LAYER_CLS = nn.LSTM
    BIDIRECTIONAL = True
