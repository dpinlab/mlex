from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split


class BaseSplitStrategy(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, timestamp_column=None):
        super().__init__()
        self.timestamp_column = timestamp_column

    @abstractmethod
    def fit(self, X, y=None):
        pass

    def transform(self, X, y):
        return X, y


class PastFutureSplit(BaseSplitStrategy):
    def __init__(self, timestamp_column=None, proportion=0.5):
        super().__init__(timestamp_column)
        self.proportion = 1 - proportion
        self.train_indices_ = None
        self.test_indices_ = None

    def fit(self, X, y=None):
        if self.timestamp_column:
            X = X.sort_values(by=[self.timestamp_column]).reset_index(drop=True)
        mid = int(self.proportion * len(X))
        self.train_indices_ = X.index[:mid]
        self.test_indices_ = X.index[mid:]

        return self

    def transform(self, X, y):
        X_train = X.loc[self.train_indices_].reset_index(drop=True)
        y_train = y.loc[self.train_indices_].reset_index(drop=True)
        X_test = X.loc[self.test_indices_].reset_index(drop=True)
        y_test = y.loc[self.test_indices_].reset_index(drop=True)
        return X_train, y_train, X_test, y_test

    def get_test_indices(self):
        return self.test_indices_


class FeatureStratifiedSplit(BaseSplitStrategy):
    def __init__(self, stratify_column=None, split_proportion=0.3, number_of_quantiles=4):
        super().__init__()
        self.stratify_column = stratify_column
        self.split_proportion = split_proportion
        self.number_of_quantiles = number_of_quantiles
        self.train_indices_ = None
        self.test_indices_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y must be provided for stratification")

        if self.stratify_column is None:
            raise ValueError("stratify_column must be provided")

        dataset_size = len(X)
        positive_counts = X[y.values == 1].groupby(self.stratify_column).size()
        total_counts = X.groupby(self.stratify_column).size()

        positive_ratio = (positive_counts / total_counts).fillna(0)

        groups_df = pd.DataFrame(total_counts).rename(columns={0: "total_samples"})
        groups_df["positive_ratio"] = positive_ratio
        groups_df["positive_ratio"] = groups_df["positive_ratio"].fillna(0)
        groups_df = groups_df.reset_index()

        groups_df["weighted_score"] = groups_df["positive_ratio"] * (groups_df["total_samples"] / dataset_size)

        min_score = groups_df["weighted_score"].min()
        max_score = groups_df["weighted_score"].max()

        if min_score == max_score:
            groups_df["cluster"] = 0
        else:
            groups_df["cluster"] = pd.qcut(
                groups_df["weighted_score"],
                q=self.number_of_quantiles,
                labels=False,
                duplicates='drop'
            )

        groups_with_positive = groups_df[groups_df['positive_ratio'] > 0]
        groups_without_positive = groups_df[groups_df['positive_ratio'] == 0]

        train_with_positive, test_with_positive = train_test_split(
            groups_with_positive[self.stratify_column], 
            test_size=self.split_proportion, 
            stratify=groups_with_positive["cluster"]
        )

        train_without_positive, test_without_positive = [], []

        if groups_without_positive.shape[0] > 1:
            train_without_positive, test_without_positive = train_test_split(
                groups_without_positive[self.stratify_column],
                test_size=0.5
            )
        elif groups_without_positive.shape[0] == 1:
            train_without_positive = groups_without_positive[self.stratify_column]

        train_groups = list(train_with_positive) + list(train_without_positive)
        test_groups = list(test_with_positive) + list(test_without_positive)

        self.train_indices_ = X[X[self.stratify_column].isin(train_groups)].index
        self.test_indices_ = X[X[self.stratify_column].isin(test_groups)].index

        return self

    def transform(self, X, y):
        return X.loc[self.train_indices_], y.loc[self.train_indices_], X.loc[self.test_indices_], y.loc[self.test_indices_]

    def get_test_indices(self):
        return self.test_indices_

    def get_groups(self, X):
        return X.loc[self.train_indices_, self.stratify_column].values, X.loc[self.test_indices_, self.stratify_column].values
