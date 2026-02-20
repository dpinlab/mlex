import sys
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..", "..")))

import torch
import numpy as np
import pandas as pd
from mlex import DataReader, MLP, F1MaxThresholdStrategy, StandardEvaluator
from pcpe_utils import get_pcpe_dtype_dict, pcpe_preprocessing_read_func

path_train = r'/data/pcpe/pcpe_03.csv'
path_test = r'/data/pcpe/pcpe_04.csv'
target_column = 'I-d'
filter_data = {'NATUREZA_LANCAMENTO': 'C'}
threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()

reader_train = DataReader(path_train, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(path_test, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
X_test, y_test = reader_test.get_X_y()

categories = [pd.unique(X_train[col]) for col in ['TIPO', 'CNAB', 'NATUREZA_SALDO']]

model_MLP = MLP(target_column='I-d', categories=categories, numeric_features=['DIA_LANCAMENTO','MES_LANCAMENTO','VALOR_TRANSACAO','VALOR_SALDO'], categorical_features=['TIPO', 'CNAB', 'NATUREZA_SALDO'])

model_MLP.fit(X_train, y_train.values.flatten())

y_pred_score = model_MLP.predict_proba(X_test)

print('\n')
evaluator = StandardEvaluator(f"MLP_pipeline", threshold_selection)
evaluator.evaluate(y_test.values.flatten(), [], y_pred_score)
print(evaluator.summary())
print('\n')

# evaluator.save('evaluation.parquet')
# evaluator.save('evaluation.json')
# model_MLP.model