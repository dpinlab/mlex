import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from abc import ABC

class BaseMLEXModule(nn.Module, ABC):
    def __init__(self, module,input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        
    def init_parameters(self):
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for module in self.model._modules.items():
            _, layers = module 
            weights = layers._all_weights
            for layer_weight in weights:
                for weight in layer_weight:
                    if 'weight_ih' in weight:
                        init.xavier_uniform_(getattr(layers,weight))
                    elif 'weight_hh' in weight:
                        init.orthogonal_(getattr(layers,weight))
                    elif 'bias' in weight:
                        init.zeros_(getattr(layers,weight))
                        
    def forward(self,x):
        hh = self.model(x)
        logits = self.linear(hh)
        return F.sigmoid(logits)

class RNNModule(BaseMLEXModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__(RNNModule, input_size, hidden_size, num_layers, num_classes)
        self.model = nn.Sequential(
        nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True),
        )
        self.init_parameters()


class LSTMModule(BaseMLEXModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__(self, RNNModule, input_size, hidden_size, num_layers, num_classes)
        self.model = nn.Sequential(
        nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True),
        )
        self.init_parameters()
        
        
class GRUModule(BaseMLEXModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__(self, RNNModule, input_size, hidden_size, num_layers, num_classes)
        self.model = nn.Sequential(
        nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True),
        )
        self.init_parameters()
        
                
        
        
if __name__ == '__main__':
    mlex_component = RNNModule(input_size=10,hidden_size=2,num_layers=2,num_classes=1)
    for p in mlex_component.parameters():
        print(p)
    