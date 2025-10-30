import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import pandas as pd
import numpy as np
import arff
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from mlex import DataReader, PreProcessingTransformer
import torch
data_path = r'/data/eeg_eyestate/EEG_Eye_State.arff'
output_dir = r'/data/eeg_eyestate'
base_filename = 'EEG_Eye_State'
output_base_path = os.path.join(output_dir, f'{base_filename}.csv')
target_column = 'eyeDetection'


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

with open(data_path, 'r') as f:
    raw = arff.load(f)

columns = [attr[0] for attr in raw['attributes']]
df = pd.DataFrame(raw['data'], columns=columns)


### Adding timestamp column

start_time = pd.Timestamp('2025-01-01 00:00:00.000')
interval = pd.to_timedelta('7.8ms')

df['Timestamp'] = [start_time + i * interval for i in range(len(df))]

#### dropping some outliers lines
df = df.drop([10386,11509, 898]) 

df['GROUP'] = 'Unknown'

df.to_csv(output_base_path, sep=';', decimal=',', index=False)
print(f"CSV base salvo: {output_base_path}")

df.drop(columns=['GROUP'], inplace=True)

reader = DataReader(output_base_path, target_columns=[target_column] )
X,y = reader.get_X_y()

numeric_cols_for_preprocessing = [col for col in X.columns.to_list() if col not in ['Timestamp','GROUP']]
preprocessor = PreProcessingTransformer(target_column=[target_column],numeric_features=numeric_cols_for_preprocessing,categorical_features=[],context_feature=['GROUP'],handle_unknown='ignore')


preprocessor.fit(X, y)

X_array, y_array = preprocessor.transform(X, y)


cluster_algorithms = {
    'kmeans': KMeans(n_clusters=3, n_init="auto", random_state=42),
    'gmm': GaussianMixture(n_components=3, random_state=42),
    'agglomerative': AgglomerativeClustering(n_clusters=3)
}

for cluster_name, cluster_model in cluster_algorithms.items():
    ####clusterização####

    if cluster_name == 'gmm':
        cluster_model.fit(X_array)
        cluster_labels = cluster_model.predict(X_array)
    else:
        cluster_labels = cluster_model.fit_predict(X_array)

    X['GROUP'] = cluster_labels

    df_clustered_output = pd.concat([X, y], axis=1)

    output_clustered_path = os.path.join(output_dir, f'{base_filename}_{cluster_name}.csv')
    
    df_clustered_output.to_csv(output_clustered_path, sep=';', decimal=',', index=False)
    print(f"CSV clusterizado com '{cluster_name}' salvo: {output_clustered_path}")

print("\nProcessamento e salvamento de todos os CSVs concluído.")