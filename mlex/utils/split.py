from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split


class BaseSplitStrategy(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, timestamp_column='DATA_LANCAMENTO'):
        super().__init__()
        self.timestamp_column = timestamp_column

    @abstractmethod
    def fit(self, X, y=None):
        pass

    def transform(self, X, y):
        return X, y


class PastFutureSplit(BaseSplitStrategy):
    def __init__(self, timestamp_column='DATA_LANCAMENTO', proportion=0.5):
        super().__init__(timestamp_column)
        self.proportion = proportion
        self.train_indices_ = None
        self.test_indices_ = None

    def fit(self, X, y=None):
        df_sorted = X.sort_values(by=[self.timestamp_column]).reset_index(drop=True)
        mid = int(self.proportion * len(df_sorted))
        train_indices = df_sorted.index[:mid]
        test_indices = df_sorted.index[mid:-1]

        train_df = df_sorted.loc[train_indices]
        test_df = df_sorted.loc[test_indices]

        # Ensure common CNAB values between splits
        common_cnab = set(train_df['CNAB']).intersection(set(test_df['CNAB']))
        self.train_indices_ = train_df[train_df['CNAB'].isin(common_cnab)].index
        self.test_indices_ = test_df[test_df['CNAB'].isin(common_cnab)].index
        return self

    def transform(self, X, y):
        return X.loc[self.train_indices_].reset_index(drop=True), y.loc[self.train_indices_].reset_index(drop=True), X.loc[self.test_indices_].reset_index(drop=True), y.loc[self.test_indices_].reset_index(drop=True)

    def get_test_indices(self):
        return self.test_indices_


class FeatureStratifiedSplit(BaseSplitStrategy):
    def __init__(self, timestamp_column='DATA_LANCAMENTO', column_to_stratify='CONTA_TITULAR', test_proportion=0.3, number_of_quantiles=4):
        super().__init__(timestamp_column)
        self.column_to_stratify = column_to_stratify
        self.test_proportion = test_proportion
        self.number_of_quantiles = number_of_quantiles
        self.train_indices_ = None
        self.test_indices_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y must be provided for stratification")

        dataset_size = len(X)
        id_counts = X[y.values == 1].groupby(self.column_to_stratify).size()
        total_counts = X.groupby(self.column_to_stratify).size()

        id_ratio = (id_counts / total_counts).fillna(0)

        accounts_df = pd.DataFrame(total_counts).rename(columns={0: "total_transactions"})
        accounts_df["id_ratio"] = id_ratio
        accounts_df["id_ratio"] = accounts_df["id_ratio"].fillna(0)
        accounts_df = accounts_df.reset_index()  # self.column_to_stratify becomes a column

        accounts_df["weighted_score"] = accounts_df["id_ratio"] * (accounts_df["total_transactions"] / dataset_size)

        min_score = accounts_df["weighted_score"].min()
        max_score = accounts_df["weighted_score"].max()

        if min_score == max_score:
            accounts_df["cluster"] = 0
        else:
            accounts_df["cluster"] = pd.qcut(
                accounts_df["weighted_score"],
                q=self.number_of_quantiles,
                labels=False,
                duplicates='drop'
            )

        train_accounts, test_accounts = train_test_split(
            accounts_df[self.column_to_stratify], 
            test_size=self.test_proportion, 
            stratify=accounts_df["cluster"]
        )

        self.train_indices_ = X[X[self.column_to_stratify].isin(train_accounts)].index
        self.test_indices_ = X[X[self.column_to_stratify].isin(test_accounts)].index

        train_df = X.loc[self.train_indices_]
        test_df = X.loc[self.test_indices_]

        # Ensure common CNAB values between splits
        #common_cnab = set(train_df['CNAB']).intersection(set(test_df['CNAB']))
        #self.train_indices_ = train_df[train_df['CNAB'].isin(common_cnab)].index
        #self.test_indices_ = test_df[test_df['CNAB'].isin(common_cnab)].index
        return self

    def transform(self, X, y):
        return X.loc[self.train_indices_], y.loc[self.train_indices_], X.loc[self.test_indices_], y.loc[self.test_indices_]

    def get_test_indices(self):
        return self.test_indices_

    def get_groups(self, X):
        return X.loc[self.train_indices_, self.column_to_stratify].values, X.loc[self.test_indices_, self.column_to_stratify].values
