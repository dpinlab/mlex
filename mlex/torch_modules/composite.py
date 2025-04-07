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
    def optimize_module(self, x, target, loss_fn=None):
        pass
    
    
class MLEXComposite(MLEXComponent):
    def __init__(self):
        super(MLEXComposite,self).__init__()
        self.loss_fn = F.binary_cross_entropy
        
    def optimize_module(self, x, target, loss_fn=None):
        for child in self.children():
            child.optimize_module(target, x, loss_fn=self.loss_fn)        
    
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
    
    def optimize_module(self,x, target, loss_fn):
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = loss_fn(output, target)
        loss.backward()
        self.optimizer.step()



if __name__ =='__main__':

    def generate_time_series(batch_size, n_steps):
        freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
        time = np.linspace(0, 1, n_steps)
        series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
        series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
        series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # + noise
        return series[..., np.newaxis].astype(np.float32)
    
    n_steps = 50
    series = generate_time_series(10000, n_steps + 1)
    X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
    X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
    X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
    
    rnn = RNNModule(input_size=1, hidden_size=3,num_layers=1,num_classes=1)
    leaf1 = MLEXLeafComponent(torch.optim.RMSprop, model=rnn)
    composite = MLEXComposite()
    composite.add_module(name='leaf1',module=leaf1)
    print(composite.forward(torch.from_numpy(X_train)))
    
    
    

