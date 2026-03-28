import pandas as pd

df = pd.read_csv("/data/pcpe/pcpe_03.csv", sep=';', decimal=',')

df5 = pd.read_csv("/data/pcpe/POSITIVE_ACCOUNTS.csv", sep=';')

df5['VALUE'] = 1

# Pivot the data
df_pivot = df4.pivot_table(
    index=['NUM_BANCO', 'NOME_BANCO', 'NUM_AGENCIA', 'NUM_CONTA'],
    columns='TIPIFICAÇÃO',
    values='VALUE',
    fill_value=0
).reset_index()

df_pivot.columns.name = None
df_pivot = df_pivot.rename_axis(None, axis=1)

df_pivot.to_csv("/data/pcpe/POSITIVE_ACCOUNTS_3.csv", sep=';', index=False)

df['I-d'] = df['I-d'].apply(lambda x: '0' if pd.isna(x) else str(int(x)))
df['CPF_CNPJ_OD'] = df['CPF_CNPJ_OD'].apply(lambda x: 'MISSING' if pd.isna(x) else x)

df_1d = df[df['I-d'] == '1']

cpf_crime = set(df_1d['CPF_CNPJ_TITULAR'].to_list())
cpf_geral = set(df['CPF_CNPJ_TITULAR'].to_list())
cpf_normal = cpf_geral - cpf_crime

print('Perc. of CPF_CNPJ_OD of I-d transactions that is missing:')
print(sum(df_1d['CPF_CNPJ_OD'] == 'MISSING') / len(df_1d['CPF_CNPJ_OD']) * 100)

# df_user = df[df['CPF_CNPJ_TITULAR'] == 'MB9177552237']
#
# print('Perc. of how many transactions of a specific user is I-d:')
# print(sum(df_user['I-d'] == '1') / len(df_user) * 100)

df_counts = df[['CPF_CNPJ_TITULAR', 'I-d']].value_counts().reset_index()
df_counts2 = df[['CPF_CNPJ_TITULAR', 'CPF_CNPJ_OD', 'I-d']].value_counts().reset_index()
# df_counts2 = df_counts2.sort_values(by=['CPF_CNPJ_TITULAR', 'I-d', 'count'], ascending=[True, False, False]).reset_index(drop=True)

cpf_cnpj_titular = set(df_1d['CPF_CNPJ_TITULAR'].to_list())
cpf_cnpj_od = set(df_1d['CPF_CNPJ_OD'].to_list())

conta_titular = set(df_1d['NUMERO_CONTA'].to_list())
conta_od = set(df_1d['NUMERO_CONTA_OD'].to_list())

anos_count = df[['ANO_LANCAMENTO']].value_counts().reset_index()
total_per_year = anos_count['count'].sum()
anos_count['percentage'] = (anos_count['count'] / total_per_year) * 100

anos_count_id = df[['ANO_LANCAMENTO', 'I-d']].value_counts().reset_index()
total_per_year = anos_count_id.groupby('ANO_LANCAMENTO')['count'].transform('sum')
anos_count_id['percentage'] = (anos_count_id['count'] / total_per_year) * 100

anos_count_ie = df[['ANO_LANCAMENTO', 'I-e']].value_counts().reset_index()
total_per_year = anos_count_ie.groupby('ANO_LANCAMENTO')['count'].transform('sum')
anos_count_ie['percentage'] = (anos_count_ie['count'] / total_per_year) * 100

anos_count_ivn = df[['ANO_LANCAMENTO', 'IV-n']].value_counts().reset_index()
total_per_year = anos_count_ivn.groupby('ANO_LANCAMENTO')['count'].transform('sum')
anos_count_ivn['percentage'] = (anos_count_ivn['count'] / total_per_year) * 100

print()