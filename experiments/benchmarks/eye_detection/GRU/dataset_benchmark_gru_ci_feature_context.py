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
from mlex import GRU
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
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
sequences_compositions = ['temporal', 'feature']
sequence_column_dict = {'temporal': None, 'feature': 'O1'}
iterations = 10
timestamp_column= 'Timestamp'
path = r'/data/eeg_eyestate/EEG_Eye_State.csv'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



reader = DataReader(path, target_columns=[target_column] )
X_full,y_full = reader.get_X_y()




for sequence_composition in sequences_compositions:
    print(f"experiment sequence {sequence_composition}")

    sequence_column = sequence_column_dict[sequence_composition]

    splitter_tt = PastFutureSplit(proportion=0.75, timestamp_column=timestamp_column)
    splitter_tt.fit(X_full, y_full)
            
    X_train_full, y_train_full, X_test, y_test = splitter_tt.transform(X_full, y_full)

    if sequence_column:
        bins = X_train_full[sequence_column].quantile(np.linspace(0, 1, 5)).unique().tolist()
            
        # Ajuste de bordas para garantir inclusão de min e max
        if len(bins) >= 2:
            bins[0] = X_train_full[sequence_column].min() - 1e-6
            bins[-1] = X_train_full[sequence_column].max() + 1e-6
            bins = sorted(list(set(bins))) # Garante ordem e unicidade
        else:
            print("Poucos valores únicos, discretização pulada.")
            continue # Pula a iteração se a discretização não for possível

        X_train_full['GROUP'] = pd.cut(
            X_train_full[sequence_column],
            bins=bins, 
            labels=False, 
            include_lowest=True,
            right=True
        ).fillna(-1).astype(int)
        
        
        X_test['GROUP'] = pd.cut(
            X_test[sequence_column],
            bins=bins,
            labels=False,
            include_lowest=True,
            right=True
        ).fillna(-1).astype(int)


        context_sorter = ContextAware(target_column=target_column, timestamp_column=timestamp_column, context_column='GROUP')
    else:
        context_sorter = ContextAware(target_column=target_column, timestamp_column=timestamp_column, context_column=None)
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
            model_GRU = GRU(validation_data=validation_data, 
                            target_column=target_column, 
                            seq_length = sequence_length, 
                            numeric_features= [col for col in X_train.columns if (col != timestamp_column and col != 'GROUP')],
                            context_feature=['GROUP'],
                            random_seed=None,
                            device=device)

            model_GRU.fit(X_train, y_train)

            y_pred_score = model_GRU.score_samples(X_test)

            y_true = model_GRU.get_y_true_sequences(X_test, y_test)

            evaluator = StandardEvaluator(f"GRU_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{sequence_length}_{sequence_composition}_{threshold_strategy}_Iteration-{i+1}", threshold_selection)
            evaluator.evaluate(np.array(y_true), [], y_pred_score)
            print(evaluator.summary())
            print('\n')
            

            evaluator.save('evaluation.parquet')
            evaluator.save('evaluation.json')

print("fim do experimento")
if __name__ == "__main__":
    print("Executando algo")