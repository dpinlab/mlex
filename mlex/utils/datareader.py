import pandas as pd
import functools
import numpy as np

class DataReader():
    def __init__(self, data_path, target_columns, filter_dict=None, dtype_dict=None, preprocessing_func=None):
        self.data_path = data_path
        self.target_columns = target_columns
        self.filter_dict = filter_dict
        self.X = None
        self.y = None

        self.dtype_dict = dtype_dict
        self.preprocessing_func = preprocessing_func

    def read_df(self):
        df = pd.read_csv(
            self.data_path,
            sep=';',
            decimal=',',
            dtype=self.dtype_dict,
            low_memory=False
        )

        df = df.loc[~df.duplicated()]

        if self.preprocessing_func:
            df = self.preprocessing_func(df)

        if self.filter_dict:
            for col, val in self.filter_dict.items():
                df = df[df[col] == val]

        return df.reset_index(drop=True)

    def get_X_y(self):
        df = self.read_df()
        self.y = df[self.target_columns]
        self.X = df.drop(columns=self.target_columns, axis=1)
        return self.X, self.y

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y


def get_pcpe_dtype_dict():
    return {
            'NUMERO_CASO': 'str',
            'NUMERO_BANCO': 'str',
            'NOME_BANCO': 'str',
            'NUMERO_AGENCIA': 'str',
            'NUMERO_CONTA': 'str',
            'TIPO': 'str',
            'CPF_CNPJ_TITULAR': 'str',
            'NOME_TITULAR': 'str',
            'DATA_LANCAMENTO': 'str',
            'CPF_CNPJ_OD': 'str',
            'NOME_PESSOA_OD': 'str',
            'CNAB': 'str',
            'DESCRICAO_LANCAMENTO': 'str',
            'VALOR_TRANSACAO': 'float64',
            'NATUREZA_LANCAMENTO': 'str',
            'I-d': 'uint8',
            'I-e': 'uint8',
            'IV-n': 'uint8',
            'RAMO_ATIVIDADE_1': 'str',
            'RAMO_ATIVIDADE_2': 'str',
            'RAMO_ATIVIDADE_3': 'str',
            'LOCAL_TRANSACAO': 'str',
            'NUMERO_DOCUMENTO': 'str',
            'NUMERO_DOCUMENTO_TRANSACAO': 'str',
            'VALOR_SALDO': 'float64',
            'NATUREZA_SALDO': 'str',
            'NUMERO_BANCO_OD': 'str',
            'NUMERO_AGENCIA_OD': 'str',
            'NUMERO_CONTA_OD': 'str',
            'NOME_ENDOSSANTE_CHEQUE': 'str',
            'DOC_ENDOSSANTE_CHEQUE': 'str',
            'DIA_LANCAMENTO': 'uint8',
            'MES_LANCAMENTO': 'uint8',
            'ANO_LANCAMENTO': 'uint16'
        }

def sorter_decorator(fn):
    @functools.wraps(fn)
    def wrapper(df, *args, **kwargs):
        df = fn(df, *args, **kwargs)
        balance_sign = np.where(df['NATUREZA_SALDO'] == 'D', -1, 1)
        tx_sign = np.where(df['NATUREZA_LANCAMENTO'] == 'D', -1, 1)
        
        df["VALOR_SALDO"] = (df["VALOR_SALDO"] * 100).round() / 100.0
        df["VALOR_TRANSACAO"] = (df["VALOR_TRANSACAO"] * 100).round() / 100.0
        
        df["VALOR_SALDO"] *= balance_sign
        df["VALOR_TRANSACAO"] *= tx_sign
        
        
        df["SALDO_ANTERIOR"] = (
            (df["VALOR_SALDO"] * 100).round() - (df["VALOR_TRANSACAO"] * 100).round()
        ) / 100.0
        df['TIMESTEP'] = pd.factorize(df['DATA_LANCAMENTO'].sort_values())[0]
        return df
    return wrapper

def pcpe_preprocessing_read_func(df):
    df['CONTA_TITULAR'] = (
                df['NUMERO_BANCO'] + '_' +
                df['NUMERO_AGENCIA'] + '_' +
                df['NUMERO_CONTA']
        )
    df['CONTA_OD'] = (
            df['NUMERO_BANCO_OD'] + '_' +
            df['NUMERO_AGENCIA_OD'] + '_' +
            df['NUMERO_CONTA_OD'].astype(str)
    )
    df['CONTA_OD'] = df['CONTA_OD'].fillna('EMPTY')
    df.loc[df['CONTA_OD'].str.contains('0_0'), 'CONTA_OD'] = 'EMPTY'

    df['DATA_LANCAMENTO'] = pd.to_datetime(df['DATA_LANCAMENTO'])
    df = df.sort_values(['DATA_LANCAMENTO']).reset_index(drop=True)

    return df