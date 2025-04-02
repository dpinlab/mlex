import torch.nn as nn
import torch.nn.functional as F

class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModule,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=num_classes)
        
    def forward(self, x):
        ht = self.rnn(x)
        z = self.output_layer
        return F.sigmoid(z)