import pandas as pd

class equal_features:
    def __init__(self) -> None:
        super().__init__()

    def match_datasets(df_train, df_test):

    
        colunas_train = list(df_train.columns)
        colunas_test = list(df_test.columns)

        # Encontrar colunas comuns
        colunas_comuns = list(set(colunas_train) & set(colunas_test))

        # Manter apenas as colunas comuns em ambos os DataFrames
        df_train = df_train[colunas_comuns]
        df_test = df_test[colunas_comuns]


        return df_train, df_test