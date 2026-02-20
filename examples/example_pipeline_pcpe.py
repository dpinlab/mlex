import sys
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..")))

import numpy as np
from mlex import DataReader, RNN, F1MaxThresholdStrategy, StandardEvaluator
from pcpe_utils import get_pcpe_dtype_dict, pcpe_preprocessing_read_func

path_train = r'/data/pcpe/pcpe_03.csv'
path_test = r'/data/pcpe/pcpe_04.csv'
target_column = 'I-d'
filter_data = {'NATUREZA_LANCAMENTO': 'C'}
sequence_composition = 'account'
sequence_column_dict = {'baseline': None, 'account': 'CONTA_TITULAR', 'individual': 'CPF_CNPJ_TITULAR'}
sequence_column = sequence_column_dict[sequence_composition]
column_to_stratify = 'CPF_CNPJ_TITULAR'
threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()

reader_train = DataReader(path_train, target_columns=[target_column], dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(path_test, target_columns=[target_column], dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
X_test, y_test = reader_test.get_X_y()

model_RNN = RNN(
    target_column='I-d',
    numeric_features=['DIA_LANCAMENTO','MES_LANCAMENTO','VALOR_TRANSACAO','VALOR_SALDO'],
    categorical_features=['TIPO', 'CNAB', 'NATUREZA_SALDO'],
    split_stratify_column=column_to_stratify,
    val_split=0.3,
    context_column=sequence_column,
    timestamp_column='DATA_LANCAMENTO',
    epochs=5,
    filter_dict=filter_data,
)

model_RNN.fit(X_train, y_train)

y_pred_score = model_RNN.predict_proba(X_test)

evaluator = StandardEvaluator(f"RNN_pipeline", threshold=model_RNN.threshold)
evaluator.evaluate(np.array(y_test.values.flatten()), [], y_pred_score)
print(evaluator.summary())
print('\n')

# evaluator.save('evaluation.parquet')
# evaluator.save('evaluation.json')
# model_RNN.model