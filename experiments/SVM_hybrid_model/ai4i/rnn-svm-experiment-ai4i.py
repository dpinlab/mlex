"""
Pipeline Completo e Unificado (Modelos Puros + Híbrido) para o AI4I 2020

Divisão dos dados (Mantendo equivalência de dados entre os experimentos)
----------------------------------------------------------------------
  path_train (ai4i2020_train.csv) -> 80 % do Dataset Original
      [Para Puros]   ├── 90 % → Treino da Rede | 10 % → Validação (Early Stopping)
      [Para Híbrido] ├── 70 % → Treino-A (Rede) | 10 % → Validação | 20 % → Treino-B (SVM)

  path_test (ai4i2020_test.csv)   -> 20 % do Dataset Original
      └── 100 % → Avaliação final de todos os pipelines        (20 % do total)
"""

import sys
from os.path import join, abspath, dirname

from itertools import product
sys.path.append(abspath(join(__file__ , "..", "..", "..", "..")))

import numpy as np
import pandas as pd
from mlex import DataReader, RNN, LSTM, GRU, hybrid_rnn_svm, StandardEvaluator
import torch
from sklearn.model_selection import train_test_split


RESULTS_DIR = dirname(__file__)
RESULTS_PARQUET = join(RESULTS_DIR, "ai4i-all_experiments_results.parquet")
RESULTS_JSON = join(RESULTS_DIR, "ai4i-all_experiments_results.json")

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
timestamp_column = 'UDI'

sequence_lengths = [10, 20, 30, 40, 50]
hidden_sizes = [10, 32]
num_layers_options = [1, 2]
iterations = 10

svm_C_grid = [0.1, 1.0, 10.0]
svm_gamma_grid = ["scale", "auto"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo selecionado para o treino das redes: {device}\n")

reader_train = DataReader(path_train, target_columns=[target_column], sep=';', quotechar='"')
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(path_test, target_columns=[target_column], sep=';', quotechar='"')
X_test, y_test = reader_test.get_X_y()


numeric_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
categorical_features = ['Type']

all_experiments = {
    'pure-rnn':  {'creator': RNN, 'params': {'val_split': 0.1}},
    'pure-lstm': {'creator': LSTM, 'params': {'val_split': 0.1}},
    'pure-gru':  {'creator': GRU, 'params': {'val_split': 0.1}},
    'hybrid':    {'creator': hybrid_rnn_svm, 'params': {
                    'rnn_train_ratio': 0.7, 
                    'rnn_val_ratio': 0.1, 
                    'svm_kernel': 'rbf'
                 }}
}

for model_key, config in all_experiments.items():
    print(f"\n==================================================")
    print(f" INICIANDO EXPERIMENTOS: {model_key.upper()} ")
    print(f"==================================================")
    
    for seq_len in sequence_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")

        if model_key == "hybrid":
            grid_values = list(product(hidden_sizes, num_layers_options, svm_C_grid, svm_gamma_grid))
        else:
            grid_values = list(product(hidden_sizes, num_layers_options))

        for hidden_size, num_layers, *extra in grid_values:
            for i in range(iterations):
                
                # Mantendo os parâmetros novos do SVM depois do Iteration para preservar seu plot
                if model_key == "hybrid":
                    svm_C, svm_gamma = extra[0], extra[1]
                    run_id = (
                        f"ai4i-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}"
                        f"_SequenceLength-{seq_len}_Iteration-{i + 1}_svmC-{svm_C}_svmGamma-{svm_gamma}"
                    )
                    grid_params = {
                        "svm_C": svm_C,
                        "svm_gamma": svm_gamma,
                    }
                else:
                    run_id = (
                        f"ai4i-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}"
                        f"_SequenceLength-{seq_len}_Iteration-{i + 1}"
                    )
                    grid_params = {}
                
                print(f"Rodando: {run_id}")

                base_params = {
                    'target_column': target_column,
                    'timestamp_column': timestamp_column,
                    'numeric_features': numeric_features,
                    'categorical_features': categorical_features,
                    'device': device,
                    'seq_length': seq_len,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'batch_size': 32,  
                    'epochs': 30,
                    'patience': 5,
                }

                full_params = {**base_params, **config['params'], **grid_params}

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

                evaluator.save(RESULTS_PARQUET)
                evaluator.save(RESULTS_JSON)
                
                del model
                del evaluator
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

print("\nExperimentos do AI4I finalizados.")