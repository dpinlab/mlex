import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SequenceDataset(Dataset):
    def __init__(self, X, y, sequence_length,column_to_stratify):
        '''
            column_to_stratify : index of column being used to group the sequences
        '''
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.column_to_stratify = column_to_stratify

        self.valid_indices = self._generate_valid_indices()


    def _generate_valid_indices(self):
        ''''
            indices[] : all the initial indexes of sequences that can be used
        '''
        indices = []
        for i in range(len(self.X) - self.sequence_length + 1):
            window = self.column_to_stratify[i:i + self.sequence_length]
            if np.all(window == window[0]):
                indices.append(i)
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        X_seq = self.X[i:i + self.sequence_length]
        y_seq = self.y[i + self.sequence_length - 1] # seq to vec

        #timestamps = X_seq[:, self.timestamps_columns]  
        #sort_idx = np.lexsort(timestamps.T)
        #X_seq_sorted = X_seq[sort_idx]  

        return X_seq, y_seq

class SequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_stratify, sequence_length=10, batch_size=32):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.column_to_stratify = column_to_stratify


    def fit(self, X, y=None):
        return self

    def transform(self, X, y):
    
        dataset = SequenceDataset(X, y, self.sequence_length, self.column_to_stratify)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
