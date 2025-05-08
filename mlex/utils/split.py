import abc
import pandas as pd
from sklearn.model_selection import train_test_split
from mlex.utils.preprocessing import PreProcessing


class BaseSplitStrategy(abc.ABC):
    def __init__(self, path, filter_dict = None, X = None, y = None) -> None:
        super().__init__()
        self.df = self.read_df(path, filter_dict)
        self.timestamp_column = 'DATA_LANCAMENTO'

    @abc.abstractmethod
    def train_test_split(self):
        pass

    def read_df(self, path, filter_dict):
        df = pd.read_csv(path, delimiter=';', decimal=',', low_memory=False)
        df = df.loc[~df.duplicated()]
        df['CONTA_TITULAR'] = df['NUMERO_BANCO'].astype(str) + '_' + df['NUMERO_AGENCIA'].astype(str) + '_' + df['NUMERO_CONTA'].astype(str)
        df['DATA_LANCAMENTO'] = pd.to_datetime(df['DATA_LANCAMENTO'])
        if filter_dict:
            for col, val in filter_dict.items():
                df = df[df[col] == val]

        return df


class PastFutureSplit(BaseSplitStrategy):
    def train_test_split(self, typology=None, proportion=.5):
        typology = ['I-d'] if typology is None else typology
        df_sorted = self.df.sort_values(by=[self.timestamp_column]).reset_index(drop=True)

        mid = int(proportion *len(df_sorted))
        train_df = df_sorted.iloc[:mid]
        test_df =  df_sorted.iloc[mid:-1] 
    
        common_values = set(train_df['CNAB']).intersection(set(test_df['CNAB']))

        train_df = train_df[train_df['CNAB'].isin(common_values)]
        test_df = test_df[test_df['CNAB'].isin(common_values)]

        train = PreProcessing(train_df)
        test = PreProcessing(test_df)

        X_train, y_train, feature_names_train = train.get_X_y(typology)
        X_test, y_test, feature_names_test = test.get_X_y(typology)

        return X_train, X_test, y_train, y_test, feature_names_train
    

    
class FeatureStratifiedSplit(BaseSplitStrategy):

    def __init__(self, path, filter_dict=None, X=None, y=None):
        super().__init__(path, filter_dict, X, y)
        self.group_test = []
        self.group_train = []

    def train_test_split(self, column_to_stratify = 'CONTA_TITULAR', typology=None, test_proportion=.3):
        typology = ['I-d'] if typology is None else typology

        dataset_size = len(self.df)
        id_counts = self.df[self.df[typology] == 1].groupby(column_to_stratify).size()
        total_counts = self.df.groupby(column_to_stratify).size()

        id_ratio = (id_counts / total_counts).fillna(0)
        accounts_df = pd.DataFrame({
            column_to_stratify: total_counts.index,
            "total_transactions": total_counts,
            "id_ratio": id_ratio
        })

        accounts_df["weighted_score"] = (
                accounts_df["id_ratio"] *
                (accounts_df["total_transactions"] / dataset_size)
        )

        accounts_df["cluster"] = pd.cut(
            accounts_df["weighted_score"],
            bins=[-1e-6, 0, 0.002, 0.02, 1.0],
            labels=[0, 1, 2, 3]
        )

        train_accounts, test_accounts = train_test_split(
            accounts_df[column_to_stratify], test_size=test_proportion, stratify=accounts_df["cluster"]
            )

        train_df = self.df[self.df[column_to_stratify].isin(train_accounts)]
        test_df = self.df[self.df[column_to_stratify].isin(test_accounts)]

        common_values = set(train_df['CNAB']).intersection(set(test_df['CNAB']))

        train_df = train_df[train_df['CNAB'].isin(common_values)]
        test_df = test_df[test_df['CNAB'].isin(common_values)]

        train_df = train_df.sort_values(by=[column_to_stratify, self.timestamp_column]).reset_index(drop=True)
        test_df = test_df.sort_values(by=[column_to_stratify, self.timestamp_column]).reset_index(drop=True)

        self.group_train = train_df[column_to_stratify].values
        self.group_test = test_df[column_to_stratify].values    

        train = PreProcessing(train_df)
        test = PreProcessing(test_df)

        X_train, y_train, feature_names_train = train.get_X_y(typology)
        X_test, y_test, feature_names_test = test.get_X_y(typology)

        return X_train, X_test, y_train, y_test, feature_names_train
