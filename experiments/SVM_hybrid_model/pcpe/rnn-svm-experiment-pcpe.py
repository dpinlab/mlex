"""
Pipeline para o modelo híbrido e baselines puras (RNN, LSTM, GRU) no PCPE

Divisão dos dados
-----------------
  path_train (pcpe_03.csv)
      [Para Puros]   ├── 90 % → Treino da Rede | 10 % → Validação (Early Stopping)
      [Para Híbrido] ├── 60 % → Treino-A (Rede) | 10 % → Validação | 30 % → Treino-B (SVM)

  path_test (pcpe_04.csv)
      └── 100 % → Avaliação final de todos os pipelines completos
"""

import sys
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..", "..", "..")))

import numpy as np
from mlex import RNN, LSTM, GRU, hybrid_rnn_svm,DataReader, F1MaxThresholdStrategy, StandardEvaluator
from pcpe_utils import get_pcpe_dtype_dict, pcpe_preprocessing_read_func
import torch

path_train = r'/data/pcpe/pcpe_03.csv'
path_test  = r'/data/pcpe/pcpe_04.csv'

target_column      = 'I-d'
filter_data        = {'NATUREZA_LANCAMENTO': 'C'}
# sequence_column    = 'CONTA_TITULAR'          
# column_to_stratify = 'CPF_CNPJ_TITULAR'
num_layers=1
hidden_size=10
iterations=10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo selecionado para o treino das redes: {device}")

reader_train = DataReader(
    path_train,
    target_columns=[target_column],
    dtype_dict=get_pcpe_dtype_dict(),
    preprocessing_func=pcpe_preprocessing_read_func,
)
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(
    path_test,
    target_columns=[target_column],
    dtype_dict=get_pcpe_dtype_dict(),
    preprocessing_func=pcpe_preprocessing_read_func,
)
X_test, y_test = reader_test.get_X_y()


# rnn_train_ratio=0.7 → 70 % do treino vai para as redes recorrentes
# rnn_val_ratio=0.1   → desses 70 %, 10% são separados para validação
# Resultado: Treino-A=60 %, Validação=10 %, Treino-B(SVM)=30 %

all_experiments = {
    'pure-rnn':  {'creator': RNN, 'params': {'val_split': 0.1}},
    'pure-lstm': {'creator': LSTM, 'params': {'val_split': 0.1}},
    'pure-gru':  {'creator': GRU, 'params': {'val_split': 0.1}},
    'hybrid':    {'creator': hybrid_rnn_svm, 'params': {'rnn_train_ratio': 0.7, 'rnn_val_ratio': 0.1, 'svm_kernel': 'rbf', 'svm_C': 1.0, 'svm_gamma': 'scale'}}
}

sequence_lengths = [10, 20, 30, 40, 50]
for model_key, config in all_experiments.items():
    print(f"\n==================================================")
    print(f" INICIANDO EXPERIMENTOS PCPE: {model_key.upper()} ")
    print(f"==================================================")
    for seq_len in sequence_lengths:
        print(f"Sequence Length: {seq_len}")
        for i in range(iterations):
            
            run_id = f"pcpe-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{seq_len}_Iteration-{i+1}"

            base_params = {
                'target_column': target_column,
                'timestamp_column': 'DATA_LANCAMENTO',
                'numeric_features': ['DIA_LANCAMENTO', 'MES_LANCAMENTO', 'VALOR_TRANSACAO', 'VALOR_SALDO'],
                'categorical_features': ['TIPO', 'NATUREZA_SALDO'],
                'filter_dict': filter_data,
                'device': device,
                'seq_length': seq_len,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'batch_size': 32,  
                'epochs': 30,
                'patience': 5,
            }

            full_params = {**base_params, **config['params']}
            model = config['creator'](**full_params)

            model.fit(X_train, y_train)

            y_pred_score = model.predict_proba(X_test)

            evaluator = StandardEvaluator(
                run_id,
                threshold=model.threshold,
            )
            evaluator.evaluate(
                np.array(y_test.values.flatten()),
                [],
                y_pred_score,
            )

            print(evaluator.summary())
            print('\n')

            evaluator.save('pcpe-all_experiments_results.parquet')
            evaluator.save('pcpe-all_experiments_results.json')

            del model
            del evaluator
            torch.cuda.empty_cache()