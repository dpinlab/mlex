import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from abc import ABC
import keras



class BaseMLEXModule(nn.Module, ABC):
    def __init__(self,module,input_size, hidden_size, num_layers, num_classes, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.module = module(input_size=self.input_size, hidden_size=self.hidden_size,num_layers=self.num_layers, batch_first=True, bidirectional=bidirectional)
        self._init_parameters()
        
    def _init_parameters(self):
        for module in self._modules.items():
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
       output, h_n = self.module(x)
       output = output[:, -1, :]
       logits = self.linear(output)
       z = self.output_layer(logits)
       return z 
                        

class RNNModel(BaseMLEXModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__(nn.RNN,input_size, hidden_size, num_layers, num_classes)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_classes)
        self.output_layer = nn.Sigmoid()
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
        
        
    
class GRUModel(BaseMLEXModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__(nn.GRU, input_size, hidden_size, num_layers, num_classes)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_classes)
        self.output_layer = nn.Sigmoid()
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)


class LSTMModel(BaseMLEXModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__(nn.LSTM, input_size, hidden_size, num_layers, num_classes)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_classes)
        self.output_layer = nn.Sigmoid()
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
       
                
        
class BILSTMModel(BaseMLEXModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__(nn.LSTM, input_size, hidden_size, num_layers, num_classes, bidirectional=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_classes)
        self.output_layer = nn.Sigmoid()
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

                
        
        
#if __name__ == '__main__':
    #mlex_component = RNNModule(input_size=10,hidden_size=2,num_layers=2,num_classes=1)
    #for p in mlex_component.parameters():
    #    print(p)
    