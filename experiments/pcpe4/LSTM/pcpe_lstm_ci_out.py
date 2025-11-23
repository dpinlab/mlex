import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import pandas as pd
import numpy as np
from mlex import LSTM, DataReader, FeatureStratifiedSplit, F1MaxThresholdStrategy, StandardEvaluator, get_pcpe_dtype_dict, pcpe_preprocessing_read_func, ContextAware, PastFutureSplit

# EXPERIMENT = "fraud-detection_nn"
# MODEL_NAME = "LSTM_baseline_Id_without_cnab"



threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()
sequence_lengths = [10, 20, 30, 40, 50]
batch_size = 32
hidden_size = 10
num_layers = 1
num_classes = 1
epochs = 30
patience = 5
target_column = 'I-d'
sequences_compositions = ['temporal', 'Feature_individual','Feature_account']
sequence_column_dict = {'temporal': None, 'Feature_individual': 'CPF_CNPJ_TITULAR', 'Feature_account': 'CONTA_TITULAR'}
iterations = 10
timestamp_column= 'DATA_LANCAMENTO'
filter_data = {'NATUREZA_LANCAMENTO': 'C'}

path_train = r'/data/pcpe/pcpe_03.csv'
path_test = r'/data/pcpe/pcpe_04.csv'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for sequence_composition in sequences_compositions:
    print(f"experiment sequence {sequence_composition}")

    colum_to_stratify = sequence_column_dict[sequence_composition]

    reader_train = DataReader(path_train, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
    X, y = reader_train.get_X_y()

    reader_test = DataReader(path_test, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
    X_test, y_test = reader_test.get_X_y()

    context_sorter = ContextAware(
            target_column=target_column, 
            timestamp_column=timestamp_column, 
            context_column=colum_to_stratify
    )

    X, y = context_sorter.transform(X.copy(), y.copy())
    X_test, y_test = context_sorter.transform(X_test.copy(), y_test.copy())

    # X['GROUP'] = 'Unknown'
    # X_test['GROUP'] = 'Unknown'

    for sequence_length in sequence_lengths:
        print(f"  sequence length: {sequence_length}")
        for i in range(iterations):
            print(f"    iteration: {i+1}/{iterations}")

            splitter_tv = FeatureStratifiedSplit(column_to_stratify='CPF_CNPJ_TITULAR', test_proportion=0.3)

            splitter_tv.fit(X, y)
            X_train, y_train, X_val, y_val = splitter_tv.transform(X, y)

            categories = [pd.unique(X_train[col]) for col in ['TIPO', 'NATUREZA_SALDO']]

            validation_data = (X_val, y_val)
            print("training model...")
            model_LSTM = LSTM(validation_data=validation_data, 
                            target_column=target_column, 
                            seq_length=sequence_length,
                            numeric_features=['DIA_LANCAMENTO','MES_LANCAMENTO','VALOR_TRANSACAO','VALOR_SALDO'], 
                            categorical_features=['TIPO', 'NATUREZA_SALDO'], 
                            context_feature=['GROUP'],
                            categories=categories, 
                            random_seed=None,
                            device=device, 
                        )

            model_LSTM.fit(X_train, y_train)

            y_pred_score = model_LSTM.score_samples(X_test)

            y_true = model_LSTM.get_y_true_sequences(X_test, y_test)

            evaluator = StandardEvaluator(f"LSTM_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{sequence_length}_{sequence_composition}_{threshold_strategy}_Iteration-{i+1}", threshold_selection)
            evaluator.evaluate(np.array(y_true), [], y_pred_score)
            print(evaluator.summary())
            print('\n')

            evaluator.save('evaluation.parquet')
            evaluator.save('evaluation.json')
print("--- Script Finished Successfully ---")