from typing import Optional, Tuple, List, Union
import pandas as pd


class ContextAware:
    def __init__(
        self,
        target_column: str,
        timestamp_column: str,
        context_column: Optional[str],
        sort_columns: Optional[List[str]] = None,
    ):
        """
        Context-aware sorting/ordering helper.

        Args:
            target_column: name of target column in y.
            timestamp_column: name of timestamp column used for ordering.
            context_column: name of column used for grouping (can be None).
            sort_columns: optional explicit list of columns to sort by.
                          If provided it will be used as-is (must exist in X after
                          CONTEXT column is added). Otherwise defaults to ['CONTEXT', timestamp_column].
        """
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.context_column = context_column
        self.sort_columns = sort_columns

    def transform(self, X: pd.DataFrame, y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Sorts the dataframe X (and optionally y) according to context and timestamp,
        or using an explicit list self.sort_columns when provided.

        Returns (X_sorted, y_sorted) or (X_sorted, None) if y is None.
        """
        df = X.copy()

        # keep original index so we can map predictions back to the original ordering later
        df['__orig_index'] = df.index.values

        # create CONTEXT column from provided context_column (or use 'Unknown')
        if self.context_column:
            df['CONTEXT'] = df[self.context_column].fillna('Unknown')
        else:
            df['CONTEXT'] = 'Unknown'

        # attach y values if provided (support Series or DataFrame)
        if y is not None:
            if isinstance(y, pd.Series):
                y_vals = y.values
            else:
                # DataFrame-like: use target_column
                y_vals = y[self.target_column].values
            df['_y'] = y_vals

        # decide sorting columns
        if self.sort_columns:
            sort_cols = list(self.sort_columns)
        else:
            sort_cols = ['CONTEXT', self.timestamp_column]

        # validate sort columns exist in df
        missing = [c for c in sort_cols if c not in df.columns]
        if missing:
            raise ValueError(f"ContextAware.sort failed — missing sort columns in X after CONTEXT: {missing}")

        # perform sort and reset index (keep __orig_index)
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

        if y is not None:
            new_y = df[['_y']].rename(columns={'_y': self.target_column})
            df.drop(columns=['_y'], inplace=True)
            return df, new_y
        else:
            return df, None
