import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataReader(BaseEstimator, TransformerMixin):
    def __init__(self, data_path, target_columns, filter_dict=None):
        self.data_path = data_path
        self.target_columns = target_columns
        self.filter_dict = filter_dict
        self.X = None
        self.y = None

        self.dtype_dict = {
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

    def read_df(self):
        df = pd.read_csv(
            self.data_path,
            sep=';',
            decimal=',',
            dtype=self.dtype_dict,
            low_memory=False
        )

        df = df.loc[~df.duplicated()]

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

        if self.filter_dict:
            for col, val in self.filter_dict.items():
                df = df[df[col] == val]

        return df.reset_index(drop=True)
    
   
    def fit(self, X=None, y=None):
        df = self.read_df()
        self.y = df[self.target_columns]
        self.X = df.drop(columns=self.target_columns, axis=1)
        return self

    def transform(self, X=None):
        return self.X

    def get_target(self):
        return self.y
