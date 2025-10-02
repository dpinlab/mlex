import sys
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..")))

import torch
import numpy as np
import pandas as pd
from mlex import DataReader, FeatureStratifiedSplit, RNN, F1MaxThresholdStrategy, StandardEvaluator

path = r'/data/pcpe/pcpe_03.csv'
target_column = 'I-d'
filter_data = {'NATUREZA_LANCAMENTO': 'C'}
sequence_composition = 'account'
sequence_column_dict = {'temporal': None, 'account': 'CONTA_TITULAR', 'individual': 'CPF_CNPJ_TITULAR'}
sequence_column = sequence_column_dict[sequence_composition]
column_to_stratify = 'CPF_CNPJ_TITULAR'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()

reader = DataReader(path, target_columns=[target_column], filter_dict=filter_data)
df = reader.read_df()

if sequence_composition != 'temporal':
    df['GROUP'] = df[sequence_column]
    df = df.sort_values(by=['GROUP', 'DATA_LANCAMENTO']).reset_index(drop=True)
else:
    df['GROUP'] = 'Global'

y = df[[target_column]]
X = df.drop(columns=[target_column], axis=1)

splitter_tt = FeatureStratifiedSplit(column_to_stratify=column_to_stratify, test_proportion=0.3)
splitter_tt.fit(X, y)
X_train, y_train, X_test, y_test = splitter_tt.transform(X, y)

splitter_tv = FeatureStratifiedSplit(column_to_stratify=column_to_stratify, test_proportion=0.3, number_of_quantiles=2)
splitter_tv.fit(X_train, y_train)
X_train, y_train, X_val, y_val = splitter_tv.transform(X_train, y_train)

categories = [pd.unique(X_train[col]) for col in ['TIPO', 'CNAB', 'NATUREZA_SALDO']]

validation_data = (X_val, y_val)
model_RNN = RNN(validation_data=validation_data, target_column='I-d', categories=categories, device=device, numeric_features=['DIA_LANCAMENTO','MES_LANCAMENTO','VALOR_TRANSACAO','VALOR_SALDO'], categorical_features=['TIPO', 'CNAB', 'NATUREZA_SALDO'], context_feature=['GROUP'])

model_RNN.fit(X_train, y_train)

y_pred_score = model_RNN.score_samples(X_test)

y_true = model_RNN.get_y_true_sequences(X_test, y_test)

evaluator = StandardEvaluator(f"RNN_pipeline", threshold_selection)
evaluator.evaluate(np.array(y_true), [], y_pred_score)
print(evaluator.summary())
print('\n')

# evaluator.save('evaluation.parquet')
# evaluator.save('evaluation.json')
# model_RNN.model