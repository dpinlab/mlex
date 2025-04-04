import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from abc import ABC, abstractmethod

class MLEXComponent(ABC, nn.Module):
    
    @abstractmethod
    def optimize_module(self, loss_fn):
        pass
    
    
class MLEXComposite(MLEXComponent):
    def __init__(self):
        super().__init__()
        self.loss_fn = F.binary_cross_entropy
        
    def optimize_modules(self):
        for child in self.children():
            child.optimize_module(self.loss_fn)
    
    
class MLEXLeafComponent(MLEXComponent):
    def __init__(self, optimizer:Optimizer, model:nn.Module):
        super().__init__()
        self.model = model
        self.optimizer = optimizer(params=model.parameters(), lr=.001, alpha=.9, eps=1e-07)
    
    def optimize_module(self, loss_fn):
        pass
        

