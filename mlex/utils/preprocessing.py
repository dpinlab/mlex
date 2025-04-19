import pandas as pd
import numpy as np
from mlex import CompositeTranformer
class PreProcessing:

    def __init__(self,df):
        self.df = df
        

    def get_X_y(self):
            
            df = self.df
            columns_num = [
                'DIA_LANCAMENTO', 
                'MES_LANCAMENTO',
                'VALOR_TRANSACAO',
                'VALOR_SALDO',
            ]

            columns_cat = [
                'TIPO',
                'CNAB',
                'NATUREZA_SALDO'
            ]
            tranformer = CompositeTranformer(
            numeric_features=columns_num,
            categorical_features=columns_cat)
            
            X = tranformer.transform(df)
            # X = df[np.concatenate([columns_num, columns_cat])].values
            target = ['I-d']
            y = df[target].values
            y = np.nan_to_num(y)
            return X, y
    
    def match_datasets(df_train, df_test):

        colunas_train = list(df_train.columns)
        colunas_test = list(df_test.columns)

        # Encontrar colunas comuns
        colunas_comuns = list(set(colunas_train) & set(colunas_test))

        # Manter apenas as colunas comuns em ambos os DataFrames
        df_train = df_train[colunas_comuns]
        df_test = df_test[colunas_comuns]


        return df_train, df_test