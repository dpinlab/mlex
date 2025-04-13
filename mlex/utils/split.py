import abc
import pandas as pd
from sklearn.model_selection import train_test_split
from mlex.utils.analysis import LacciAnalysis


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
    def train_test_split(self, proportion=.5):
        mid = int(proportion *len(self.X))
        X_train = self.X[:mid]
        y_train = self.y[:mid]
        X_test = self.X[mid:-1]
        y_test = self.y[mid:-1]
        return X_train, X_test, y_train, y_test
    
class AccountStratifiedSplit(BaseSplitStrategy):

    def train_test_split(self, typology='I-d', proportion=.3):

        id_counts = self.df[self.df[typology] == 1].groupby('CONTA_TITULAR').size()
        total_counts = self.df.groupby('CONTA_TITULAR').size()
        id_ratio = (id_counts / total_counts).fillna(0)

        accounts_df = pd.DataFrame({"CONTA_TITULAR": total_counts.index, "id_ratio": id_ratio})
        accounts_df["cluster"] = pd.cut(accounts_df["id_ratio"], bins=[-0.01, 0, 0.25, 0.5, 1.0], labels=[0, 1, 2, 3])

        train_accounts, test_accounts = train_test_split(
            accounts_df["CONTA_TITULAR"], test_size=proportion, stratify=accounts_df["cluster"]
            )

        train_df = self.df[self.df['CONTA_TITULAR'].isin(train_accounts)]
        test_df = self.df[self.df['CONTA_TITULAR'].isin(test_accounts)]

        common_values = set(train_df['CNAB']).intersection(set(test_df['CNAB']))

        train_df = train_df[train_df['CNAB'].isin(common_values)]
        test_df = test_df[test_df['CNAB'].isin(common_values)]

        lacci_analysis_train = LacciAnalysis(train_df)
        lacci_analysis_test = LacciAnalysis(test_df)

        X_train, y_train, feature_names_train = lacci_analysis_train.get_X_y()
        X_test, y_test, feature_names_test = lacci_analysis_test.get_X_y()

        return X_train, X_test, train_df['CONTA_TITULAR'], y_train, y_test, feature_names_train
    
class AccountStratifiedSplit:
    def train_test_split(self, typology='I-d', proportion=.3):

        id_counts = self.df[self.df[typology] == 1].groupby('CONTA_TITULAR').size()
        total_counts = self.df.groupby('CONTA_TITULAR').size()
        id_ratio = (id_counts / total_counts).fillna(0)

        accounts_df = pd.DataFrame({"CONTA_TITULAR": total_counts.index, "id_ratio": id_ratio})
        accounts_df["cluster"] = pd.cut(accounts_df["id_ratio"], bins=[-0.01, 0, 0.25, 0.5, 1.0], labels=[0, 1, 2, 3])

        train_accounts, test_accounts = train_test_split(
            accounts_df["CONTA_TITULAR"], test_size=proportion, stratify=accounts_df["cluster"]
            )

        train_df = self.df[self.df['CONTA_TITULAR'].isin(train_accounts)]
        test_df = self.df[self.df['CONTA_TITULAR'].isin(test_accounts)]

        common_values = set(train_df['CNAB']).intersection(set(test_df['CNAB']))

        train_df = train_df[train_df['CNAB'].isin(common_values)]
        test_df = test_df[test_df['CNAB'].isin(common_values)]

        lacci_analysis_train = LacciAnalysis(train_df)
        lacci_analysis_test = LacciAnalysis(test_df)

        X_train, y_train, feature_names_train = lacci_analysis_train.get_X_y()
        X_test, y_test, feature_names_test = lacci_analysis_test.get_X_y()

        return X_train, X_test, train_df['CONTA_TITULAR'], y_train, y_test, feature_names_train


class CpfStratifiedSplit:
    def train_test_split(self, typology='I-d', proportion=.3):

        id_counts = self.df[self.df[typology] == 1].groupby('CPF_CNPJ_TITULAR').size()
        total_counts = self.df.groupby('CPF_CNPJ_TITULAR').size()
        id_ratio = (id_counts / total_counts).fillna(0)

        accounts_df = pd.DataFrame({"CPF_CNPJ_TITULAR": total_counts.index, "id_ratio": id_ratio})
        accounts_df["cluster"] = pd.cut(accounts_df["id_ratio"], bins=[-0.01, 0, 0.25, 0.5, 1.0], labels=[0, 1, 2, 3])

        train_accounts, test_accounts = train_test_split(
            accounts_df["CPF_CNPJ_TITULAR"], test_size=proportion, stratify=accounts_df["cluster"]
            )

        train_df = self.df[self.df['CPF_CNPJ_TITULAR'].isin(train_accounts)]
        test_df = self.df[self.df['CPF_CNPJ_TITULAR'].isin(test_accounts)]

        common_values = set(train_df['CNAB']).intersection(set(test_df['CNAB']))

        train_df = train_df[train_df['CNAB'].isin(common_values)]
        test_df = test_df[test_df['CNAB'].isin(common_values)]

        lacci_analysis_train = LacciAnalysis(train_df)
        lacci_analysis_test = LacciAnalysis(test_df)

        X_train, y_train, feature_names_train = lacci_analysis_train.get_X_y()
        X_test, y_test, feature_names_test = lacci_analysis_test.get_X_y()

        return X_train, X_test, train_df['CPF_CNPJ_TITULAR'], y_train, y_test, feature_names_train