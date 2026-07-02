import sys, os
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..", "..", "..")))

import numpy as np
import pandas as pd
import torch
from mlex import RNN, LSTM, GRU, hybrid_rnn_svm, DataReader, StandardEvaluator

path_train = r'/data/occupancy/datatraining.txt'
path_test  = r'/data/occupancy/datatest.txt'

dataset_name = "occupancy"
target_column = 'Occupancy'

num_layers = 1
hidden_size = 10
iterations = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo selecionado para o treino: {device}")


reader_train = DataReader(
    path_train,
    target_columns=[target_column],
    sep=",",          
    quotechar='"',
    # Seus dados usam separação por vírgula no txt/csv do UCI
)
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(
    path_test,
    target_columns=[target_column],
    sep=",",          # Garante a separação correta por vírgula
    quotechar='"',
)
X_test, y_test = reader_test.get_X_y()

# =========================================================================== #
# ARQUITETURAS DOS EXPERIMENTOS
# =========================================================================== #
all_experiments = {
    'pure-rnn':  {'creator': RNN, 'params': {'val_split': 0.1}},
    'pure-lstm': {'creator': LSTM, 'params': {'val_split': 0.1}},
    'pure-gru':  {'creator': GRU, 'params': {'val_split': 0.1}},
    'hybrid':    {'creator': hybrid_rnn_svm, 'params': {
                    'rnn_train_ratio': 0.7, 
                    'rnn_val_ratio': 0.1, 
                    'svm_kernel': 'rbf', 
                    'svm_C': 1.0, 
                    'svm_gamma': 'scale'
                 }}
}

# Janelas temporais de minutos para avaliar no escritório
sequence_lengths = [10, 20, 30, 40, 50]

for model_key, config in all_experiments.items():
    print(f"\n==================================================")
    print(f" INICIANDO EXPERIMENTOS {dataset_name.upper()}: {model_key.upper()} ")
    print(f"==================================================")
    
    for seq_len in sequence_lengths:
        print(f"Sequence Length: {seq_len}")
        for i in range(iterations):
            
            # Formato de run_id idêntico ao que o nosso script de plotagem espera ler
            run_id = f"{dataset_name}-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{seq_len}_Iteration-{i+1}"

            base_params = {
                'target_column': target_column,
                'timestamp_column': 'date', # Coluna temporal nativa do UCI Occupancy
                'numeric_features': ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'],
                'categorical_features': [], # Este dataset não possui colunas categóricas adicionais
                'filter_dict': {}, # Sem filtros estáticos iniciais
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

            # Treinamento
            model.fit(X_train, y_train)

            # Predição
            y_pred_score = model.predict_proba(X_test)

            # Avaliação com salvamento padronizado
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

            # Salvando os resultados na estrutura mapeada pelo script de plotagem unificado
            output_file = f"{dataset_name}-all_experiments_results"
            evaluator.save(f"{output_file}.parquet")
            evaluator.save(f"{output_file}.json")

            del model
            del evaluator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

print(f"\n[FIM] Todos os experimentos do {dataset_name.upper()} foram executados com sucesso!")