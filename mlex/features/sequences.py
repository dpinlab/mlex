import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SequenceDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_seq = self.y[idx + self.sequence_length - 1]  
        return X_seq, y_seq

class SequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_length=10, batch_size=32):
        self.sequence_length = sequence_length
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y):
        dataset = SequenceDataset(X, y, self.sequence_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader
