import torch
import tensorflow as tf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from abc import ABC, abstractmethod
from mlex.models.models import (RNNModule, LSTMModule, GRUModule)
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
            child.optimize_module(x, target, loss_fn=self.loss_fn)        
    
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
        print(loss)
        loss.backward()
        self.optimizer.step()



if __name__ =='__main__':

    def generate_sequence_classification_data(num_sequences=10, seq_length=20, input_dim=1, noise_level=0.1):
        X = np.zeros((num_sequences, seq_length, input_dim))
        y = np.zeros((num_sequences, seq_length, 1))

        for i in range(num_sequences):
            # Randomly decide if this sequence will be positive (1) or negative (0)
            sequence_class = np.random.randint(0, 2)

            # Create a base signal that's different for positive vs negative sequences
            if sequence_class == 1:
                # Positive sequence pattern (e.g., increasing trend)
                base_signal = np.linspace(0, 1, seq_length)
            else:
                # Negative sequence pattern (e.g., decreasing trend)
                base_signal = np.linspace(1, 0, seq_length)

            # Add some noise
            noise = noise_level * np.random.randn(seq_length)
            signal = base_signal + noise

            # Store the sequence and labels
            X[i, :, 0] = signal  # Assuming input_dim=1
            y[i, :, 0] = sequence_class  # Same label for all timesteps

        # Convert to PyTorch tensors with correct dtype
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        return X_tensor, y_tensor
    
    X_train, y_train = generate_sequence_classification_data(num_sequences=1000, seq_length=20)
    X_val, y_val = generate_sequence_classification_data(num_sequences=200, seq_length=20)
    X_test, y_test = generate_sequence_classification_data(num_sequences=200, seq_length=20)
    rnn = RNNModule(input_size=1, hidden_size=3,num_layers=1,num_classes=1)
    lstm = LSTMModule(input_size=1, hidden_size=3,num_layers=1,num_classes=1)
    gru = GRUModule(input_size=1, hidden_size=3,num_layers=1,num_classes=1)
    leaf1 = MLEXLeafComponent(torch.optim.RMSprop, model=rnn)
    leaf2 = MLEXLeafComponent(torch.optim.RMSprop, model=lstm)
    leaf3 = MLEXLeafComponent(torch.optim.RMSprop, model=gru)
    composite = MLEXComposite()
    data = X_train
    target = y_train    
    composite.add_module(name='leaf1',module=leaf1)
    composite.add_module(name='leaf2',module=leaf2)
    composite.add_module(name='leaf3',module=leaf3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    composite.to(device=device)
    #print(len(composite.forward(data)))
    composite.optimize_module(x=data, target=target)
    
    

