import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import torch
from mlex import DataReader
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from mlex import PreProcessingTransformer
import pandas as pd


clusters = ['kmeans', 'gmm', 'agglomerative']
path = r'/data/isa/EEG_Eye_State_com_timestamp.arff'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

target_column = 'eyeDetection'
filter_data = {}

reader = DataReader(path, target_columns=[target_column], filter_dict=filter_data)
X = reader.fit_transform(X=None)
y = reader.get_target()

######
#retirando timestamp
X = X.drop(columns=["Timestamp"])

#####

X = X.drop([10386,11509,898]) 
y = y.drop([10386,11509,898]) 


print(f'COLUNAS DO X : {X.columns}')
preprocessor = PreProcessingTransformer(target_columns=[target_column],numeric_features=X.columns,categorical_features=[], handle_unknown='ignore')

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
    
    estados = sorted(clusters_series.unique())
    num_estados = len(estados)
    
    mapeamento_estados = {estado: i for i, estado in enumerate(estados)}
    
    transicoes_frequencia = pd.DataFrame(
        0, 
        index=estados, 
        columns=estados, 
        dtype=int
    )
    
    for i in range(len(clusters_series) - 1):
        estado_atual = clusters_series.iloc[i]
        proximo_estado = clusters_series.iloc[i+1]
        transicoes_frequencia.loc[estado_atual, proximo_estado] += 1
        
    print(f"\n--- Matriz de Frequência de Transição para o algoritmo: {cluster_name.upper()} ---")
    print(transicoes_frequencia)

    # 2. Calcula a Matriz de Transição de Probabilidade
    # A soma de cada linha representa o total de transições a partir daquele estado
    soma_linha = transicoes_frequencia.sum(axis=1)
    
    # Evita divisão por zero
    soma_linha[soma_linha == 0] = 1 
    
    # Divide cada linha pela sua soma para obter as probabilidades
    matriz_probabilidade = transicoes_frequencia.div(soma_linha, axis=0)
    
    print(f"\n--- Matriz de Probabilidade de Transição para o algoritmo: {cluster_name.upper()} ---")
    print(matriz_probabilidade)

    print("\n" + "="*80)

    