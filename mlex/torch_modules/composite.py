import torch
import tensorflow as tf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from abc import ABC, abstractmethod
from models import RNNModule
class MLEXComponent(ABC, nn.Module):
    
    @abstractmethod
    def optimize_module(self):
        pass
    
    
class MLEXComposite(MLEXComponent):
    def __init__(self):
        super(MLEXComposite,self).__init__()
        self.loss_fn = F.binary_cross_entropy
        
    def optimize_module(self):
        for child in self.children():
            child.optimize_module(self.loss_fn)
    
    def forward(self,x):
        results = []
        for child in self.children():
            results.append(child.forward(x))
        return results
    
class MLEXLeafComponent(MLEXComponent):
    def __init__(self, optimizer, model:nn.Module):
        super().__init__()
        self.model = model
        self.optimizer = optimizer(params=self.parameters(), lr=.001, alpha=.9, eps=1e-07)
    
    def forward(self, x):
        return self.model.forward(x)
    
    def parameters(self):
        return self.model.parameters()
    
    def optimize_module(self, loss_fn):
        pass


if __name__ =='__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    torch.manual_seed(42)
    rnn = RNNModule(input_size=3, hidden_size=3,num_layers=1,num_classes=1)
    leaf1 = MLEXLeafComponent(torch.optim.RMSprop, model=rnn)
    composite = MLEXComposite()
    composite.add_module(name='leaf1',module=leaf1)
    print(composite.leaf1)
    print(leaf1.parameters())
    for p in composite.parameters():
        print(p)
    
    
    

