import sys
import os
from os.path import join, abspath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import numpy as np
from mlex import DataReader, RNN, LSTM, GRU, BILSTM, F1MaxThresholdStrategy, StandardEvaluator
from mlex.evaluation.plotter import EvaluationPlotter
from pcpe_utils import get_pcpe_dtype_dict, pcpe_preprocessing_read_func
import matplotlib.pyplot as plt


current_dir = os.path.dirname(os.path.abspath(__file__))

data_base_path = os.path.abspath(os.path.join(current_dir, "../../../../"))

path_train = os.path.join(data_base_path, 'pcpe_03_drive.csv')
path_test = os.path.join(data_base_path, 'pcpe_04_drive.csv')

target_column = 'I-d'
timestamp_column = 'DATA_LANCAMENTO'
filter_data = {'NATUREZA_LANCAMENTO': 'C'}
# sequence_composition = 'account'
sequences_compositions = ['baseline', 'account', 'individual']
sequence_column_dict = {'baseline': None, 'account': 'CONTA_TITULAR', 'individual': 'CPF_CNPJ_TITULAR'}
# sequence_column = sequence_column_dict[sequence_composition]
column_to_stratify = 'CPF_CNPJ_TITULAR'
threshold_strategy = 'f1max'

output_parquet = "evaluation_results_full.parquet"

model_classes = [RNN, LSTM, GRU, BILSTM]

threshold_selection = F1MaxThresholdStrategy()

########
batch_size = 32
hidden_size = 10
num_layers = 1
num_classes = 1
epochs = 30
patience = 5
iterations = 10
sequence_lengths = [10, 20, 30, 40, 50]
#######



reader_train = DataReader(path_train, target_columns=[target_column],filter_dict=filter_data ,dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(path_test, target_columns=[target_column],filter_dict=filter_data ,dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
X_test, y_test = reader_test.get_X_y()


for ModelClass in model_classes:
    model_name = ModelClass.__name__
    print(f"Running experiments for model: {model_name}")
    for seq_len in sequence_lengths:
        print(f"Sequence Length: {seq_len}")
        for comp_name in sequences_compositions:
            print(f"Sequence Composition: {comp_name}")
            context_col = sequence_column_dict[comp_name]

            for i in range(iterations):
                run_id = f"{model_name}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{seq_len}_{comp_name}_{threshold_strategy}_Iteration-{i+1}"
    

                model = ModelClass(
                    target_column=target_column,
                    timestamp_column=timestamp_column,
                    context_column=context_col,
                    split_stratify_column=column_to_stratify,
                    val_split=0.3,
                    filter_dict=filter_data,
                    numeric_features=['DIA_LANCAMENTO','MES_LANCAMENTO','VALOR_TRANSACAO','VALOR_SALDO'],
                    categorical_features=['TIPO', 'NATUREZA_SALDO'],
                    seq_length=seq_len,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_size=batch_size,
                    epochs=epochs,
                    patience=patience,
                    
                )

                model.fit(X_train, y_train)

                y_pred_score = model.predict_proba(X_test)

                # evaluator = StandardEvaluator(run_id, threshold=threshold_selection)
                evaluator = StandardEvaluator(run_id, threshold=model.threshold)
                evaluator.evaluate(np.array(y_test.values.flatten()), [], y_pred_score)
                print(evaluator.summary())
                print('\n')
 
                evaluator.save('evaluation_results_full.parquet')
                evaluator.save('evaluation_results_full.json')
# model_RNN.model

print("\n--- Treinamento Finalizado. Iniciando Plots ---")