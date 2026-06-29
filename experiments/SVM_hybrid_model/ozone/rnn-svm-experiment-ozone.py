"""
Pipeline para o modelo híbrido e baselines puras (RNN, LSTM, GRU) no UCI Ozone Level Detection

Divisão dos dados (Padronizado com o experimento AI4I 2020)
----------------------------------------------------------
  path_original -> Separado cronologicamente (PastFutureSplit: Teste = 20%)
  
  Fatia de Treino (80% do total):
      [Para Puros]   ├── 90 % → Treino da Rede | 10 % → Validação (Early Stopping)
      [Para Híbrido] ├── 60 % → Treino-A (Rede) | 10 % → Validação | 30 % → Treino-B (SVM)

  Fatia de Teste (20% do total):
      └── 100 % → Avaliação final de todos os pipelines (os 20% de dias mais recentes)
"""

import sys
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..", "..", "..")))

import numpy as np
import pandas as pd
import torch
from ucimlrepo import fetch_ucirepo

from mlex import RNN, LSTM, GRU, hybrid_rnn_svm, DataReader, StandardEvaluator
from mlex.utils.split import PastFutureSplit

UCI_DATASET_ID = 172
DATASET_SUBSET = "8hr"
TARGET_COLUMN = "Class"
TIMESTAMP_COLUMN = "Date"
CATEGORICAL_FEATURES = []

NUM_LAYERS = 1
HIDDEN_SIZE = 10
ITERATIONS = 10  
TEST_PROPORTION = 0.2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo selecionado para o treino das redes: {DEVICE}\n")

# --------------------------------------------------------------------------- #
# 1. Carregamento e Pré-processamento dos Dados (Inspirado no código UCI)
# --------------------------------------------------------------------------- #
print(f"Buscando dataset UCI id={UCI_DATASET_ID} (Ozone Level Detection)...")
ds = fetch_ucirepo(id=UCI_DATASET_ID)

features = ds.data.features.reset_index(drop=True)
targets = ds.data.targets.reset_index(drop=True)
ids = ds.data.ids.reset_index(drop=True)

# Filtrando para o subset de 8 horas conforme o pipeline padrão do dataset
mask = ids["Dataset"] == DATASET_SUBSET
print(f"  Filtrando para o subset '{DATASET_SUBSET}': {int(mask.sum()):,} de {len(ids):,} linhas")

X = features.loc[mask].copy()
y = targets.loc[mask].copy()
dates = pd.to_datetime(ids.loc[mask, "Date"], format="%m/%d/%Y")

X[TIMESTAMP_COLUMN] = dates.values
X = X.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
y = y.loc[X.index].reset_index(drop=True)

# Remover linhas sem target e converter dados faltantes (NaN) pela mediana
valid = ~y[TARGET_COLUMN].isna()
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)
y[TARGET_COLUMN] = y[TARGET_COLUMN].astype(int)

numeric_features = [c for c in X.columns if c != TIMESTAMP_COLUMN]
X[numeric_features] = X[numeric_features].apply(pd.to_numeric, errors="coerce")
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median(numeric_only=True))

# Divisão Cronológica (Passado / Futuro)
past_future = PastFutureSplit(timestamp_column=TIMESTAMP_COLUMN, proportion=TEST_PROPORTION)
past_future.fit(X, y)
X_train, y_train, X_test, y_test = past_future.transform(X, y)

print(f"  Divisão Concluída — Treino: {len(X_train):,} | Teste: {len(X_test):,}")
print(f"  Taxa de Positivos — Treino: {y_train[TARGET_COLUMN].mean():.4f} | Teste: {y_test[TARGET_COLUMN].mean():.4f}\n")

# --------------------------------------------------------------------------- #
# 2. Loop de Experimentos Unificado
# --------------------------------------------------------------------------- #
all_experiments = {
    'pure-rnn':  {'creator': RNN, 'params': {'val_split': 0.1}},
    'pure-lstm': {'creator': LSTM, 'params': {'val_split': 0.1}},
    'pure-gru':  {'creator': GRU, 'params': {'val_split': 0.1}},
    'hybrid':    {'creator': hybrid_rnn_svm, 'params': {'rnn_train_ratio': 0.7, 'rnn_val_ratio': 0.1, 'svm_kernel': 'rbf', 'svm_C': 1.0, 'svm_gamma': 'scale'}}
}

sequence_lengths = [10, 20, 30, 40, 50]

for model_key, config in all_experiments.items():
    print(f"==================================================")
    print(f" INICIANDO EXPERIMENTOS OZONE: {model_key.upper()} ")
    print(f"==================================================")
    
    for seq_len in sequence_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        for i in range(ITERATIONS):
            
            run_id = f"ozone-{model_key}_Layers-{NUM_LAYERS}_HiddenSize-{HIDDEN_SIZE}_SequenceLength-{seq_len}_Iteration-{i+1}"
            print(f"Rodando: {run_id}")

            # Parâmetros base compartilhados do dataset Ozone
            base_params = {
                'target_column': TARGET_COLUMN,
                'timestamp_column': TIMESTAMP_COLUMN,
                'numeric_features': numeric_features,
                'categorical_features': CATEGORICAL_FEATURES,
                'device': DEVICE,
                'seq_length': seq_len,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'batch_size': 32,  
                'epochs': 30,
                'patience': 5,
            }

            # Fusão dos parâmetros gerais com os específicos de split de cada modelo
            full_params = {**base_params, **config['params']}
            
            # Instanciação dinâmica da arquitetura corrente
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

            # Salvando resultados unificados do dataset Ozone
            evaluator.save('ozone-all_experiments_results.parquet')
            evaluator.save('ozone-all_experiments_results.json')
            
            # Gerenciamento de memória agressivo para prevenir vazamentos na GPU (OOM)
            del model
            del evaluator
            torch.cuda.empty_cache()