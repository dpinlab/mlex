import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SequenceDataset(Dataset):
    def __init__(self, X, y=None, sequence_length=None, column_to_stratify_index=None):
        '''
            column_to_stratify : index of column being used to group the sequences
        '''
        self.X = torch.tensor(np.delete(X, column_to_stratify_index, axis=1).astype(np.float32), dtype=torch.float32)
        self.y = y
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.column_to_stratify = X[:, column_to_stratify_index]

        if self.column_to_stratify is not None:
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
        if self.y is not None:
            y_seq = self.y[i + self.sequence_length - 1] # seq to vec
            return X_seq, y_seq

        return X_seq


class SequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_length=10, batch_size=32, shuffled=True, column_to_stratify_index=None):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffled = shuffled
        self.column_to_stratify_index = column_to_stratify_index

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        dataset = SequenceDataset(X, y, self.sequence_length, self.column_to_stratify_index)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffled)
        return dataloader


# class SequenceTransformer2(BaseEstimator, TransformerMixin):
#     def __init__(self, sequence_length=10, group_column_index=None):
#         self.sequence_length = sequence_length
#         self.group_column_index = group_column_index

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X = np.asarray(X)
#         if y is not None:
#             y = np.asarray(y)

#         # If grouping, use the group column to ensure all in a sequence are from the same group
#         if self.group_column_index is not None:
#             group_col = X[:, self.group_column_index]
#             X = np.delete(X, self.group_column_index, axis=1)
#         else:
#             group_col = None

#         n_samples, n_features = X.shape

#         X_seqs = []
#         y_seqs = []

#         for i in range(n_samples - self.sequence_length + 1):
#             if group_col is not None:
#                 window = group_col[i:i + self.sequence_length]
#                 if not np.all(window == window[0]):
#                     continue  # skip if not all in the same group
#             X_seq = X[i:i + self.sequence_length]
#             y_seq = y[i + self.sequence_length - 1] if y is not None else None
#             X_seqs.append(X_seq)
#             if y is not None:
#                 y_seqs.append(y_seq)

#         X_seqs = np.stack(X_seqs) if X_seqs else np.empty((0, self.sequence_length, n_features))
#         y_seqs = np.array(y_seqs) if y is not None else None

#         return (X_seqs, y_seqs) if y is not None else X_seqs