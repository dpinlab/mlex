import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import torch
from mlex import DataReader
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from mlex import PreProcessingTransformer, MarkovAnalyzer, MarkovAnalyzerPlotter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


clusters = ['kmeans', 'gmm', 'agglomerative']
path = r'/data/eeg_eyestate/EEG_Eye_State.csv'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
script_path = Path(__file__).parent.absolute().joinpath('markov_results')

target_column = 'eyeDetection'

reader = DataReader(path, target_columns=[target_column])
X,y = reader.get_X_y()

######
#retirando timestamp
X = X.drop(columns=["Timestamp"])

#####



print(f'COLUNAS DO X : {X.columns}')
preprocessor = PreProcessingTransformer(target_column=[target_column],numeric_features=X.columns.to_list(),categorical_features=[], handle_unknown='ignore')

X['GROUP'] = 'Unknown'
preprocessor.fit(X, y)

X_array, y_array = preprocessor.transform(X, y)


# preprocessor = PreProcessingTransformer(target_columns=[target_column],numeric_features=X.columns,categorical_features=[], handle_unknown='ignore')
# preprocessor.fit(X, y)

# X_array = preprocessor.transform(X, y)

cluster_algorithms = {
    'kmeans': KMeans(n_clusters=3, n_init="auto", random_state=42),
    'gmm': GaussianMixture(n_components=3, random_state=42),
    'agglomerative': AgglomerativeClustering(n_clusters=3)
}

for cluster_name, cluster_model in cluster_algorithms.items():
    ####clusterização####

    if cluster_name == 'gmm':
        # cluster_model.fit(X_array)
        # cluster_labels = cluster_model.predict(X_array)
        cluster_model.fit(X_array)
        cluster_labels = cluster_model.predict(X_array)
    else:
        # cluster_labels = cluster_model.fit_predict(X_array)
        cluster_labels = cluster_model.fit_predict(X_array)

    X['cluster'] = cluster_labels

    
    
    clusters_series = X['cluster']
    Markov_Analyzer = MarkovAnalyzer()
    frequencies, probability_matrix = Markov_Analyzer.analyze(array_data=clusters_series, column_name=cluster_name, path_save=script_path)
    
    Markov_Analyzer_Plotter = MarkovAnalyzerPlotter()
    Markov_Analyzer_Plotter.plot_markov_chain(probability_matrix,  column_name=cluster_name, path_save=script_path)
    