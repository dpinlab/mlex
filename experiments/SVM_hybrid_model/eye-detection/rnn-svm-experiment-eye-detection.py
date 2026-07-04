"""
Pipeline Completo e Unificado (Modelos Puros + Híbrido) para o Eye Detection

Divisão dos dados
-----------------
  path_train (eye_detection_train_80.csv) -> treino (80%, sendo 10% para validação)
        [Para Puros]   ├── 90 % → Treino da Rede | 10 % → Validação (Early Stopping)
      [Para Híbrido] ├── 70 % → Treino-A (Rede) | 10 % → Validação | 20 % → Treino-B (SVM)
  path_test  (eye_detection_test_20.csv) -> avaliação final (20%)

Eye Detection com grid search sobre:
  - hidden_size: [10, 32]
  - num_layers: [1, 2]
  - hybrid SVM: C [0.1, 1.0, 10.0], gamma ['scale', 'auto']
"""

import sys
from itertools import product
from os.path import abspath, dirname, join

sys.path.append(abspath(join(__file__, "..", "..", "..", "..")))

import numpy as np
import pandas as pd
import torch

from mlex import DataReader, GRU, LSTM, RNN, StandardEvaluator, hybrid_rnn_svm


RESULTS_DIR = dirname(__file__)
RESULTS_PARQUET = join(RESULTS_DIR, "eye-detection-all_experiments_results.parquet")
RESULTS_JSON = join(RESULTS_DIR, "eye-detection-all_experiments_results.json")

path_train = r"/data/eeg_eyestate/eye_detection_train_80.csv"
path_test = r"/data/eeg_eyestate/eye_detection_test_20.csv"

target_column = "eyeDetection"
timestamp_column = "timestamp"

sequence_lengths = [10, 20, 30, 40, 50]
hidden_sizes = [10, 32]
num_layers_options = [1, 2]
iterations = 10

# Grid do SVM usado no pipeline híbrido
svm_C_grid = [0.1, 1.0, 10.0]
svm_gamma_grid = ["scale", "auto"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo selecionado para o treino das redes: {device}\n")

reader_train = DataReader(path_train, target_columns=[target_column], sep=',', quotechar='"')
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(path_test, target_columns=[target_column], sep=',', quotechar='"')
X_test, y_test = reader_test.get_X_y()


numeric_features = [
    col for col in X_train.columns
    if col not in {target_column, timestamp_column, "GROUP"}
]
categorical_features = []

all_experiments = {
    "pure-rnn": {
        "creator": RNN,
        "params": {"val_split": 0.1},
    },
    "pure-lstm": {
        "creator": LSTM,
        "params": {"val_split": 0.1},
    },
    "pure-gru": {
        "creator": GRU,
        "params": {"val_split": 0.1},
    },
    "hybrid": {
        "creator": hybrid_rnn_svm,
        "params": {
            "rnn_train_ratio": 0.7,
            "rnn_val_ratio": 0.1,
            "svm_kernel": "rbf",
        },
    },
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
                if model_key == "hybrid":
                    svm_C, svm_gamma = extra[0], extra[1]
                    run_id = (
                        f"eye-detection-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}"
                        f"_SequenceLength-{seq_len}_Iteration-{i + 1}_svmC-{svm_C}_svmGamma-{svm_gamma}"
                    )
                    grid_params = {
                        "svm_C": svm_C,
                        "svm_gamma": svm_gamma,
                    }
                else:
                    run_id = (
                        f"eye-detection-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}"
                        f"_SequenceLength-{seq_len}_Iteration-{i + 1}"
                    )
                    grid_params = {}
                print(f"Rodando: {run_id}")

                base_params = {
                    "target_column": target_column,
                    "timestamp_column": timestamp_column,
                    "numeric_features": numeric_features,
                    "categorical_features": categorical_features,
                    "device": device,
                    "seq_length": seq_len,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "batch_size": 32,
                    "epochs": 30,
                    "patience": 5,
                }

                full_params = {**base_params, **config["params"], **grid_params}

                model = config["creator"](**full_params)
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
                torch.cuda.empty_cache()

print("\nExperimentos finalizados.")
