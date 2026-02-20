import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List, Tuple, Union


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequential data with optional grouping.
    
    This dataset creates sliding windows of sequences from input data.
    If a group column is specified, sequences are constrained to stay
    within the same group (e.g., user sessions, time series per entity).
    """

    def __init__(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        sequence_length: int = None,
        group_column_index: Optional[int] = None,
        cache_tensors: bool = False
    ):
        """
        Initialize SequenceDataset.
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) (optional)
            sequence_length: Length of each sequence window
            group_column_index: Column index used to group sequences (e.g., user_id, session_id)
            cache_tensors: Whether to cache tensor conversions (useful for repeated access)
        
        Raises:
            ValueError: If sequence_length is None or data length < sequence_length
        """
        if sequence_length is None:
            raise ValueError("sequence_length is required")
        
        self.sequence_length = sequence_length
        self.group_column_index = group_column_index
        self.cache_tensors = cache_tensors
        self._cache = {} if cache_tensors else None

        X = np.asarray(X)
        self._validate_inputs(X, y)

        self.group_column = X[:, group_column_index] if group_column_index is not None else None
    
        self.X = self._extract_features(X, group_column_index)
        self.y = self._convert_targets(y)
        
        self.valid_indices, self.valid_end_indices = self._generate_valid_indices()
    
    def _validate_inputs(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        if len(X) < self.sequence_length:
            raise ValueError(
                f"Data length ({len(X)}) must be >= sequence_length ({self.sequence_length})"
            )
        
        if y is not None and len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
            )
    
    def _extract_features(self, X: np.ndarray, exclude_column: Optional[int]) -> torch.Tensor:
        if exclude_column is not None:
            X_features = np.delete(X, exclude_column, axis=1)
        else:
            X_features = X
        
        if X_features.size == 0:
            return torch.empty((len(X), 0), dtype=torch.float32)

        try:
            return torch.tensor(X_features, dtype=torch.float32)
        except (ValueError, TypeError):
            return self._convert_mixed_types(X_features)
    
    def _convert_mixed_types(self, X: np.ndarray) -> torch.Tensor:
        numeric_data = []
        for col_idx in range(X.shape[1]):
            try:
                numeric_data.append(X[:, col_idx].astype(np.float32))
            except (ValueError, TypeError):
                continue
        
        if not numeric_data:
            return torch.empty((X.shape[0], 0), dtype=torch.float32)
        
        return torch.tensor(np.column_stack(numeric_data), dtype=torch.float32)
    
    def _convert_targets(self, y: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        return torch.tensor(y, dtype=torch.float32) if y is not None else None
    
    def _generate_valid_indices(self) -> List[int]:
        if self.group_column is None:
            return self._generate_ungrouped_indices()
        return self._generate_grouped_indices()
    
    def _generate_ungrouped_indices(self) -> List[int]:
        max_start_idx = len(self.X) - self.sequence_length
        min_end_idx = self.sequence_length - 1
        return list(range(max_start_idx + 1)), list(range(min_end_idx, len(self.X)))
    
    def _generate_grouped_indices(self) -> List[int]:
        valid_indices = []
        valid_end_indices = []
        i = 0
        max_idx = len(self.X) - self.sequence_length + 1
        
        while i < max_idx:
            end_idx = i + self.sequence_length
            window_groups = self.group_column[i:end_idx]

            if self._is_homogeneous_group(window_groups):
                valid_indices.append(i)
                valid_end_indices.append(end_idx - 1)
                i += 1
            else:
                i += self._find_next_group_boundary(window_groups)
        
        return valid_indices, valid_end_indices
    
    def _is_homogeneous_group(self, window: np.ndarray) -> bool:
        return np.all(window == window[0])
    
    def _find_next_group_boundary(self, window: np.ndarray) -> int:
        first_value = window[0]
        change_positions = np.where(window != first_value)[0]
        return int(change_positions[0]) if len(change_positions) > 0 else 1
    
    def _get_sequence(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        X_seq = self.X[start_idx:end_idx]
        
        if self.y is not None:
            y_value = self.y[end_idx - 1]  # Target at sequence end
            return X_seq, y_value
        
        return X_seq
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get sequence at index.
        
        Args:
            idx: Index in the dataset (0 to len(dataset)-1)
        
        Returns:
            If y exists: (X_sequence, y_value) where y_value is the target at sequence end
            Otherwise: X_sequence only
        """
        if self.cache_tensors and idx in self._cache:
            return self._cache[idx]
        
        result = self._get_sequence(idx)
        
        if self.cache_tensors:
            self._cache[idx] = result
        
        return result
    
    def clear_cache(self) -> None:
        """Clear cached tensors to free memory."""
        if self._cache is not None:
            self._cache.clear()


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