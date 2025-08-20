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
from mlex import RNNModel
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering



threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()
sequence_lengths = [10, 20, 30, 40, 50]
batch_size = 32
column_to_stratify = 'cluster'
hidden_size = 10
num_layers = 1
num_classes = 1
epochs = 30
patience = 5
target_column = 'eyeDetection'
filter_data = {}
sequences_compositions = ['feature', 'temporal']
sequence_column_dict = {'temporal': None, 'feature': 'cluster'}
iterations = 10
path = r'/data/isa/EEG_Eye_State_com_timestamp.arff'



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



reader = DataReader(path, target_columns=[target_column], filter_dict=filter_data)
X_full = reader.fit_transform(X=None)
y_full = reader.get_target()

######
#retirando timestamp
X_full = X_full.drop(columns=["Timestamp"])

#####

X_full = X_full.drop([10386,11509,898]) 
y_full = y_full.drop([10386,11509,898]) 


cluster_algorithms = {
    'kmeans': KMeans(n_clusters=3, n_init="auto", random_state=42),
    'gmm': GaussianMixture(n_components=3, random_state=42),
    'agglomerative': AgglomerativeClustering(n_clusters=3)
}


for cluster_name, cluster_model in cluster_algorithms.items():
    ####clusterização####

    if cluster_name == 'gmm':
        cluster_model.fit(X_full)
        cluster_labels = cluster_model.predict(X_full)
    else:
        cluster_labels = cluster_model.fit_predict(X_array)
    

    X_full['cluster'] = cluster_labels
    y_full[target_column] = y_full[target_column].astype(int)

    splitter_tt = PastFutureSplit(proportion=0.75)
    splitter_tt.fit(X_full, y_full)
               
    X_train_full, y_train_full, X_test, y_test = splitter_tt.transform(X_full, y_full)

    ########
    print(f"\n\n### Rodando experimento com clusterização: {cluster_name} ###\n")

    for sequence_composition in sequences_compositions:
        print(f"experiment sequence {sequence_composition}")
        sequence_column = sequence_column_dict[sequence_composition]
        for sequence_length in sequence_lengths:

            if sequence_composition != 'temporal':
                X_train_full['GROUP'] = X_train_full[sequence_column].fillna('Unknown')
                X_test['GROUP'] = X_test[sequence_column].fillna('Unknown')
            else:
                X_train_full['GROUP'] = 'Unknown'
                X_test['GROUP'] = 'Unknown'

        
            splitter_tv = FeatureStratifiedSplit(column_to_stratify=column_to_stratify, test_proportion=0.3, number_of_quantiles=2)
            splitter_tv.fit(X_train_full, y_train_full)
            X_train, y_train, X_val, y_val = splitter_tv.transform(X_train_full, y_train_full)

            validation_data = (X_val, y_val)
            model_RNN = RNN(validation_data=validation_data, target_column=target_column, seq_length = sequence_length, device=device)

            model_RNN.fit(X_train, y_train)

            y_pred_score = model_RNN.score_samples(X_test)

            y_true = model_RNN.get_y_true_sequences(X_test, y_test)

            evaluator = StandardEvaluator(f"RNN_SequenceLength-{sequence_length}_{sequence_composition}_{threshold_strategy}", threshold_selection)
            evaluator.evaluate(np.array(y_true), [], y_pred_score)
            print(evaluator.summary())
            print('\n')
            

            evaluator.save('evaluation.parquet')
            evaluator.save('evaluation.json')

print("fim do experimento")
if __name__ == "__main__":
    print("Executando algo")
