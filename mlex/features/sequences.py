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

        if column_to_stratify is not None:
            self.valid_indices = self._generate_valid_indices()
        else:
            self.valid_indices = np.arange(len(X) - self.sequence_length + 1).tolist()


    def _generate_valid_indices(self):
        ''''
            indices[] : all the initial indexes of sequences that can be used
        '''
        indices = []
        i = 0
        while i < (len(self.X) - self.sequence_length + 1):
            window = self.column_to_stratify[i:i + self.sequence_length]
            if np.all(window == window[0]):
                indices.append(i)
                i += 1
            else:
                i += np.min(np.where(np.array(window) != window[0]))
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        X_seq = self.X[i:i + self.sequence_length]
        y_seq = self.y[i + self.sequence_length - 1] # seq to vec

        return X_seq, y_seq


class SequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_length=10, batch_size=32, shuffled=True):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffled = shuffled

    def fit(self, X, y=None):
        return self

    def transform(self, X, y, column_to_stratify):
        dataset = SequenceDataset(X, y, self.sequence_length, column_to_stratify)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffled)
        return dataloader
