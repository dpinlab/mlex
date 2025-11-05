import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from copy import deepcopy
from mlex import SequenceTransformer
from mlex import FeatureStratifiedSplit, PastFutureSplit
from mlex import PreProcessingTransformer
from mlex import DataReader
from mlex import StandardEvaluator
from mlex import F1MaxThresholdStrategy
from mlex import RNN
from mlex import ContextAware



threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()
sequence_lengths = [10, 20, 30, 40, 50]
batch_size = 32
hidden_size = 10
num_layers = 1
num_classes = 1
epochs = 30
patience = 5
target_column = 'eyeDetection'
sequences_compositions = ['feature', 'temporal']
sequence_column_dict = {'temporal': None, 'feature': 'GROUP'}
iterations = 10
timestamp_column= 'Timestamp'
base_path = r'/data/eeg_eyestate/EEG_Eye_State'
cluster_names = ['kmeans', 'gmm', 'agglomerative']

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


for cluster_name in cluster_names:
    
    train_path = f'{base_path}_{cluster_name}_train.csv'
    test_path = f'{base_path}_{cluster_name}_test.csv'
    print(f"\n\n### Rodando experimento: {cluster_name} (Lendo: {os.path.basename(train_path)} e {os.path.basename(test_path)}) ###\n")
    
    reader_train = DataReader(train_path, target_columns=[target_column] )
    X_train_full, y_train_full = reader_train.get_X_y()
    
    reader_test = DataReader(test_path, target_columns=[target_column] )
    X_test, y_test = reader_test.get_X_y()


    for sequence_composition in sequences_compositions:
        print(f"experiment sequence {sequence_composition}")
        sequence_column = sequence_column_dict[sequence_composition]
        
        context_sorter = ContextAware(
            target_column=target_column, 
            timestamp_column=timestamp_column, 
            context_column=sequence_column
        )

        ####### ordenacao por contexto

        X_train_full, y_train_full = context_sorter.transform(X_train_full.copy(), y_train_full.copy())
        X_test, y_test = context_sorter.transform(X_test.copy(), y_test.copy())
        ################

        for sequence_length in sequence_lengths:
            for i in range(iterations):
                            
                splitter_tv = PastFutureSplit(proportion=0.66, timestamp_column=timestamp_column)
                splitter_tv.fit(X_train_full, y_train_full)
                X_train, y_train, X_val, y_val = splitter_tv.transform(X_train_full, y_train_full)
            
                validation_data = (X_val, y_val)
                model_RNN = RNN(validation_data=validation_data, 
                                target_column=target_column, 
                                seq_length = sequence_length, 
                                numeric_features= [col for col in X_train.columns if (col != timestamp_column and col != 'GROUP')],
                                context_feature=['GROUP'],
                                random_seed=None,
                                device=device
                            )

                model_RNN.fit(X_train, y_train)

                y_pred_score = model_RNN.score_samples(X_test)

                y_true = model_RNN.get_y_true_sequences(X_test, y_test)

                evaluator = StandardEvaluator(f"RNN_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{sequence_length}_{sequence_composition}_{threshold_strategy}_Iteration-{i+1}_{cluster_name}", threshold_selection)
                evaluator.evaluate(np.array(y_true), [], y_pred_score)
                print(evaluator.summary())
                print('\n')
                

                evaluator.save('evaluation.parquet')
                evaluator.save('evaluation.json')

print("fim do experimento")
if __name__ == "__main__":
    print("Executando algo")
