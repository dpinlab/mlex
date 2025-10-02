import pandas as pd
import numpy as np
import openpyxl

df = pd.read_excel(r'/data/pcpe/v4/EXTRATO-NOVO-2-A4I4Q8 - SEM OBSERVACAO.xlsx', dtype=str)

dtype_dict = {
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

df.drop(columns=['OBSERVACAO'], axis=1, inplace=True)

df.rename(columns={'RAMO ATIVIDADE 1': 'RAMO_ATIVIDADE_1',
                   'RAMO ATIVIDADE 2': 'RAMO_ATIVIDADE_2',
                   'RAMO ATIVIDADE 3': 'RAMO_ATIVIDADE_3',
                   'I - D = DEP FRAC': 'I-d',
                   'I - E = SAQ FRAC': 'I-e',
                   'IV - N = DIV REG': 'IV-n'}, inplace=True)

df['ANO_LANCAMENTO'] = df['DATA_LANCAMENTO'].str[:4].astype(np.uint16)
df['MES_LANCAMENTO'] = df['DATA_LANCAMENTO'].str[5:7].astype(np.uint8)
df['DIA_LANCAMENTO'] = df['DATA_LANCAMENTO'].str[8:10].astype(np.uint8)

df = df[[col for col in dtype_dict.keys()]]

df.to_csv(r'/data/pcpe/pcpe_04.csv', sep=';', decimal=',', index=False)