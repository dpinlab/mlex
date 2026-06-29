"""
Pipeline Completo e Unificado (Modelos Puros + Híbrido) para o AI4I 2020

Divisão dos dados (Mantendo equivalência de dados entre os experimentos)
----------------------------------------------------------------------
  path_train (ai4i2020_train.csv) -> 80 % do Dataset Original
      [Para Puros]   ├── 90 % → Treino da Rede | 10 % → Validação (Early Stopping)
      [Para Híbrido] ├── 60 % → Treino-A (Rede) | 10 % → Validação | 30 % → Treino-B (SVM)

  path_test (ai4i2020_test.csv)   -> 20 % do Dataset Original
      └── 100 % → Avaliação final de todos os pipelines        (20 % do total)
"""

import sys
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..", "..", "..")))

import numpy as np
import pandas as pd
from mlex import DataReader, RNN, LSTM, GRU, hybrid_rnn_svm, StandardEvaluator
import torch
from sklearn.model_selection import train_test_split

path_original = r'/data/ai4i-2020-predictive-maintenance/ai4i2020.csv'
path_train = r'/data/ai4i-2020-predictive-maintenance/ai4i2020_train.csv'
path_test  = r'/data/ai4i-2020-predictive-maintenance/ai4i2020_test.csv'

# ETL e tratamento de Data Leakage
df = pd.read_csv(path_original)
leakage_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df = df.drop(columns=leakage_columns)

df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
df_train.to_csv(path_train, index=False, sep=';')
df_test.to_csv(path_test, index=False, sep=';')
print(f"Treino: {len(df_train)} linhas | Teste: {len(df_test)} linhas.\n")

target_column = 'Machine failure'
num_layers = 1
hidden_size = 10
iterations = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo selecionado para o treino das redes: {device}\n")

reader_train = DataReader(path_train, target_columns=[target_column])
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(path_test, target_columns=[target_column])
X_test, y_test = reader_test.get_X_y()

all_experiments = {
    'pure-rnn':  {'creator': RNN, 'params': {'val_split': 0.1}},
    'pure-lstm': {'creator': LSTM, 'params': {'val_split': 0.1}},
    'pure-gru':  {'creator': GRU, 'params': {'val_split': 0.1}},
    'hybrid':    {'creator': hybrid_rnn_svm, 'params': {'rnn_train_ratio': 0.7, 'rnn_val_ratio': 0.1, 'svm_kernel': 'rbf', 'svm_C': 1.0, 'svm_gamma': 'scale'}}
}

sequence_lengths = [10, 20, 30, 40, 50]

for model_key, config in all_experiments.items():
    print(f"\n==================================================")
    print(f" INICIANDO EXPERIMENTOS: {model_key.upper()} ")
    print(f"==================================================")
    
    for seq_len in sequence_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        for i in range(iterations):
            
            run_id = f"ai4i-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{seq_len}_Iteration-{i+1}"
            print(f"Rodando: {run_id}")

            # Parâmetros base compartilhados por todas as arquiteturas
            base_params = {
                'target_column': target_column,
                'timestamp_column': 'UDI',
                'numeric_features': ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'],
                'categorical_features': ['Type'],
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

            evaluator.save('ai4i-all_experiments_results.parquet')
            evaluator.save('ai4i-all_experiments_results.json')
            
            del model
            del evaluator
            torch.cuda.empty_cache()
           