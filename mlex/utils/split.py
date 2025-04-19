import abc
import pandas as pd
from sklearn.model_selection import train_test_split
from mlex.utils.preprocessing import PreProcessing


class BaseSplitStrategy(abc.ABC):
    def __init__(self, path, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.df = pd.read_csv(path, delimiter=';')

    @abc.abstractmethod
    def train_test_split(self):
        pass


class PastFutureSplit(BaseSplitStrategy):
    def train_test_split(self,timestamps_columns,proportion=.5):

        sorted_indices = self.df.sort_values(by=timestamps_columns).index
        X_sorted = self.X.loc[sorted_indices].reset_index(drop=True)
        y_sorted = self.y.loc[sorted_indices].reset_index(drop=True)

        mid = int(proportion *len(self.X))
        X_train = X_sorted[:mid]
        y_train = y_sorted[:mid]
        X_test = X_sorted[mid:-1]
        y_test = y_sorted[mid:-1]
        return X_train, X_test, y_train, y_test
    
class FeatureStratifiedSplit(BaseSplitStrategy):
    def train_test_split(self, column_to_stratify = 'CONTA_TITULAR', typology='I-d', test_proportion=.3):

        id_counts = self.df[self.df[typology] == 1].groupby(column_to_stratify).size()
        total_counts = self.df.groupby(column_to_stratify).size()
        id_ratio = (id_counts / total_counts).fillna(0)

        accounts_df = pd.DataFrame({f"{column_to_stratify}": total_counts.index, "id_ratio": id_ratio})
        accounts_df["cluster"] = pd.cut(accounts_df["id_ratio"], bins=[-0.01, 0, 0.25, 0.5, 1.0], labels=[0, 1, 2, 3])

        train_accounts, test_accounts = train_test_split(
            accounts_df[column_to_stratify], test_size=test_proportion, stratify=accounts_df["cluster"]
            )

        train_df = self.df[self.df[column_to_stratify].isin(train_accounts)]
        test_df = self.df[self.df[column_to_stratify].isin(test_accounts)]

        common_values = set(train_df['CNAB']).intersection(set(test_df['CNAB']))

        train_df = train_df[train_df['CNAB'].isin(common_values)]
        test_df = test_df[test_df['CNAB'].isin(common_values)]

        train_df = train_df.sort_values(by=column_to_stratify).reset_index(drop=True)
        test_df = test_df.sort_values(by=column_to_stratify).reset_index(drop=True)

        lacci_analysis_train = PreProcessing(train_df)
        lacci_analysis_test = PreProcessing(test_df)

        X_train, y_train = lacci_analysis_train.get_X_y()
        X_test, y_test = lacci_analysis_test.get_X_y()

        return X_train, X_test, y_train, y_test

