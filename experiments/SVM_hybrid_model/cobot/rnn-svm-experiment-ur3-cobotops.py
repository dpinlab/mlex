import sys
from itertools import product
from os.path import join, abspath, dirname
sys.path.append(abspath(join(__file__ , "..", "..", "..", "..")))

import numpy as np
import pandas as pd
import torch

from mlex import DataReader,RNN, LSTM, GRU, hybrid_rnn_svm, StandardEvaluator

"""
Pipeline Completo e Unificado (Modelos Puros + Híbrido) para o UR3 CobotOps

Divisão dos dados (Mantendo equivalência de dados entre os experimentos)
----------------------------------------------------------------------
  path_train (cobotops_train_80.csv) -> 80 % do Dataset Original
      [Para Puros]   ├── 90 % → Treino da Rede | 10 % → Validação (Early Stopping)
      [Para Híbrido] ├── 70 % → Treino-A (Rede) | 10 % → Validação | 20 % → Treino-B (SVM)

  path_test (cobotops_test_20.csv)   -> 20 % do Dataset Original
      └── 100 % → Avaliação final de todos os pipelines        (20 % do total)
"""

RESULTS_DIR = dirname(__file__)
RESULTS_PARQUET = join(RESULTS_DIR, "cobot-all_experiments_results.parquet")
RESULTS_JSON = join(RESULTS_DIR, "cobot-all_experiments_results.json")

# Caminhos locais apontando para os novos CSVs gerados pelo script de preparação
PATH_TRAIN = r'/data/ur3-cobotops/cobotops_train_80.csv'
PATH_TEST  = r'/data/ur3-cobotops/cobotops_test_20.csv'

TARGET_COLUMN = "Robot_ProtectiveStop"
TIMESTAMP_COLUMN = "Timestamp"
CATEGORICAL_FEATURES = []

SEQUENCE_LENGTHS = [10, 20, 30, 40, 50]
HIDDEN_SIZES = [10, 32]
NUM_LAYERS_OPTIONS = [1, 2]
ITERATIONS = 10  

SVM_C_GRID = [0.1, 1.0, 10.0]
SVM_GAMMA_GRID = ["scale", "auto"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo selecionado para o treino das redes: {DEVICE}\n")

# --------------------------------------------------------------------------- #
# 1. Carregamento das Bases Já Tratadas e Binarizadas
# --------------------------------------------------------------------------- #

reader_train = DataReader(PATH_TRAIN, target_columns=[TARGET_COLUMN], sep=';', quotechar='"')
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(PATH_TEST, target_columns=[TARGET_COLUMN], sep=';', quotechar='"')
X_test, y_test = reader_test.get_X_y()

# Mapeando as features numéricas restantes (garantindo que o timestamp permaneça)
numeric_features = [c for c in X_train.columns if c not in  {TARGET_COLUMN, TIMESTAMP_COLUMN}]

print(f"  Bases Carregadas — Treino: {len(X_train):,} | Teste: {len(X_test):,}")
print(f"  Quantidade de Positivos — Treino: {y_train[TARGET_COLUMN].sum():,} | Teste: {y_test[TARGET_COLUMN].sum():,}\n")
# --------------------------------------------------------------------------- #
# 2. Loop de Experimentos Unificado
# --------------------------------------------------------------------------- #
all_experiments = {
    'pure-rnn':  {'creator': RNN, 'params': {'val_split': 0.1}},
    'pure-lstm': {'creator': LSTM, 'params': {'val_split': 0.1}},
    'pure-gru':  {'creator': GRU, 'params': {'val_split': 0.1}},
    'hybrid':    {'creator': hybrid_rnn_svm, 'params': {
                    'rnn_train_ratio': 0.7,   # Mantido 70% para consistência com o AI4I e Occupancy
                    'rnn_val_ratio': 0.1,     
                    'svm_kernel': 'rbf'
                 }}
}

for model_key, config in all_experiments.items():
    print(f"==================================================")
    print(f" INICIANDO EXPERIMENTOS COBOT: {model_key.upper()} ")
    print(f"==================================================")
    
    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n--- Sequence Length: {seq_len} ---")

        # Define a combinação do Grid dinamicamente conforme a arquitetura
        if model_key == "hybrid":
            grid_values = list(product(HIDDEN_SIZES, NUM_LAYERS_OPTIONS, SVM_C_GRID, SVM_GAMMA_GRID))
        else:
            grid_values = list(product(HIDDEN_SIZES, NUM_LAYERS_OPTIONS))

        for hidden_size, num_layers, *extra in grid_values:
            for i in range(ITERATIONS):
                
                # Montando o run_id padronizado para manter compatibilidade com seus scripts de plot
                if model_key == "hybrid":
                    svm_C, svm_gamma = extra[0], extra[1]
                    run_id = (
                        f"cobot-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}"
                        f"_SequenceLength-{seq_len}_Iteration-{i + 1}_svmC-{svm_C}_svmGamma-{svm_gamma}"
                    )
                    grid_params = {
                        "svm_C": svm_C,
                        "svm_gamma": svm_gamma,
                    }
                else:
                    run_id = (
                        f"cobot-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}"
                        f"_SequenceLength-{seq_len}_Iteration-{i + 1}"
                    )
                    grid_params = {}
                
                print(f"Rodando: {run_id}")

                base_params = {
                    'target_column': TARGET_COLUMN,
                    'timestamp_column': TIMESTAMP_COLUMN,
                    'numeric_features': numeric_features,
                    'categorical_features': CATEGORICAL_FEATURES,
                    'device': DEVICE,
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

                # Salvando resultados unificados usando caminhos absolutos seguros
                evaluator.save(RESULTS_PARQUET)
                evaluator.save(RESULTS_JSON)
                
                del model
                del evaluator
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

print("\n[FIM] Todos os experimentos do UR3 CobotOps local foram finalizados.")