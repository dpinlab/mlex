import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from mlex.features.sequences import SequenceDataset


class SequenceSpanCalculator(BaseEstimator, TransformerMixin):
    """
    Transformer to compute spans for sequences.
    """

    def __init__(self, sequence_length, group_column, time_column, time_diff_fn=None):
        self.sequence_length = sequence_length
        self.group_column = group_column
        self.time_column = time_column
        self.time_diff_fn = time_diff_fn
        self.spans_ = None

    def _vectorized_span_calculation(self, df_sorted, valid_indices):
        start_indices = np.array(valid_indices)
        end_indices = start_indices + self.sequence_length - 1

        valid_mask = end_indices < len(df_sorted)
        valid_starts = start_indices[valid_mask]
        valid_ends = end_indices[valid_mask]

        start_times = df_sorted[self.time_column].iloc[valid_starts].values
        end_times = df_sorted[self.time_column].iloc[valid_ends].values

        if self.time_diff_fn:
            spans = self.time_diff_fn(end_times, start_times)
        else:
            spans = end_times - start_times

        return spans.tolist()

    def fit(self, X: pd.DataFrame, y=None):
        df_sorted = X.sort_values([self.group_column, self.time_column]).reset_index(drop=True)
        X_group = df_sorted[[self.group_column]].values

        dataset = SequenceDataset(X=X_group, y=None,
                                  sequence_length=self.sequence_length,
                                  column_to_stratify_index=0)

        self.spans_ = self._vectorized_span_calculation(df_sorted, dataset.valid_indices)
        return self

    def transform(self, X):
        return self.spans_


class SequenceSpanAnalyzer(BaseEstimator, TransformerMixin):
    """
    Analyze spans across sequence lengths and compositions.
    """

    def __init__(self, sequence_lengths, group_column, time_column, time_diff_fn=None, summary_bins=None):
        self.sequence_lengths = sequence_lengths
        self.group_column = group_column
        self.time_column = time_column
        self.time_diff_fn = time_diff_fn
        self.summary_bins = summary_bins or [
            ('span_0', lambda x: x == 0),
            ('span_1', lambda x: x == 1),
            ('span_1_7', lambda x: 1 <= x <= 7),
            ('span_1_30', lambda x: 1 <= x <= 30),
            ('span_1_90', lambda x: 1 <= x <= 90),
            ('span_1_180', lambda x: 1 <= x <= 180),
            ('span_1_365', lambda x: 1 <= x <= 365),
            ('span_over_365', lambda x: x > 365),
        ]
        self.results_ = {}

    def fit(self, X_dict: dict, y=None):
        for composition, df in tqdm(X_dict.items(), desc="Processing Compositions"):
            comp_results = {}
            for seq_len in tqdm(self.sequence_lengths, desc=f"Testing {composition}", leave=False):
                calc = SequenceSpanCalculator(
                    sequence_length=seq_len,
                    group_column=self.group_column,
                    time_column=self.time_column,
                    time_diff_fn=self.time_diff_fn
                )
                spans = calc.fit_transform(df)

                if spans:
                    arr = np.array(spans)
                    comp_results[seq_len] = {
                        'spans': spans,
                        'mean': np.mean(arr),
                        'median': np.median(arr),
                        'mode': stats.mode(arr, keepdims=True).mode[0],
                        'std': np.std(arr),
                        'min': np.min(arr),
                        'max': np.max(arr),
                        'num_sequences': len(arr),
                        **{label: sum(fn(x) for x in arr) for label, fn in self.summary_bins}
                    }
                else:
                    comp_results[seq_len] = {
                        'spans': [],
                        'mean': 0, 'median': 0, 'mode': 0,
                        'std': 0, 'min': 0, 'max': 0,
                        'num_sequences': 0,
                        **{label: 0 for label, _ in self.summary_bins}
                    }
            self.results_[composition] = comp_results
        return self

    def transform(self, X):
        return self.results_




def create_summary_table(results: dict) -> pd.DataFrame:
    """Convert results dict into summary DataFrame."""
    summary_data = []

    first_metrics = next(iter(next(iter(results.values())).values()))
    bin_labels = [k for k in first_metrics.keys() if k not in (
        'spans', 'mean', 'median', 'mode', 'std', 'min', 'max', 'num_sequences'
    )]

    for composition, comp_results in results.items():
        for seq_len, metrics in comp_results.items():
            row = {
                'Composition': composition,
                'Sequence_Length': seq_len,
                'Num_Sequences': metrics['num_sequences'],
                'Mean': round(metrics['mean'], 2),
                'Median': round(metrics['median'], 2),
                'Mode': metrics['mode'],
                'Std': round(metrics['std'], 2),
                'Min': metrics['min'],
                'Max': metrics['max'],
            }
            for label in bin_labels:
                row[f'{label}_pct'] = round(metrics[label] / max(metrics['num_sequences'], 1) * 100, 2)
            summary_data.append(row)

    return pd.DataFrame(summary_data)