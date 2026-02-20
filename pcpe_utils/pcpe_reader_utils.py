import pandas as pd

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
