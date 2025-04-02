import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

class BaseMLEXModule(nn.Module, ABC):
    def __init__(self, module,input_size, hidden_size, num_layers, num_classes):
        super(module,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

    def forward(self,x):
        ht = self.model(x)
        z = self.output_layer
        return F.sigmoid(z)

class RNNModule(BaseMLEXModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__(self, RNNModule, input_size, hidden_size, num_layers, num_classes)
        self.model = nn.Sequential(
        nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True),
        nn.Linear(in_features=hidden_size, out_features=num_classes)
        )


class LSTMModule(BaseMLEXModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__(self, RNNModule, input_size, hidden_size, num_layers, num_classes)
        self.model = nn.Sequential(
        nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True),
        nn.Linear(in_features=hidden_size, out_features=num_classes)
        )