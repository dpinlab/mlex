import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import pandas as pd
import numpy as np
import arff
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from mlex import DataReader, PreProcessingTransformer, ContextAware,PastFutureSplit
import torch
data_path = r'/data/eeg_eyestate/EEG_Eye_State.arff'
output_dir = r'/data/eeg_eyestate'
base_filename = 'EEG_Eye_State_inverted'
output_base_path = os.path.join(output_dir, f'{base_filename}.csv')
target_column = 'eyeDetection'
timestamp_column= 'Timestamp'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

with open(data_path, 'r') as f:
    raw = arff.load(f)

columns = [attr[0] for attr in raw['attributes']]
df = pd.DataFrame(raw['data'], columns=columns)

######### invertendo y ########
df[target_column] = df[target_column].astype(int)
df[target_column] = 1 - df[target_column]

################################


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


#### split #####
splitter_tt = PastFutureSplit(proportion=0.75, timestamp_column=timestamp_column)
splitter_tt.fit(X, y)

X_train, y_train, X_test, y_test = splitter_tt.transform(X, y)

###################


numeric_cols_for_preprocessing = [col for col in X.columns.to_list() if col not in ['Timestamp','GROUP']]
preprocessor = PreProcessingTransformer(target_column=[target_column],numeric_features=numeric_cols_for_preprocessing,categorical_features=[],context_feature=['GROUP'],handle_unknown='ignore')


preprocessor.fit(X_train, y_train)
X_train_array, _ = preprocessor.transform(X_train, y_train)


X_test_array, _ = preprocessor.transform(X_test, y_test)

def get_cluster_model(cluster_name):
    if cluster_name == 'kmeans':
        return KMeans(n_clusters=3, n_init="auto", random_state=42)
    elif cluster_name == 'gmm':
        return GaussianMixture(n_components=3, random_state=42)
    elif cluster_name == 'agglomerative':
        return AgglomerativeClustering(n_clusters=3)


cluster_names = ['kmeans', 'gmm', 'agglomerative']

for cluster_name in cluster_names:
    print(f"\n--- Processando {cluster_name} ---")
    cluster_model_train = get_cluster_model(cluster_name) 

    if cluster_name == 'gmm':
        cluster_model_train.fit(X_train_array)
        train_cluster_labels = cluster_model_train.predict(X_train_array)
    else:
        train_cluster_labels = cluster_model_train.fit_predict(X_train_array)

    
    cluster_model_test = get_cluster_model(cluster_name) 
    
    if cluster_name == 'gmm':
        cluster_model_test.fit(X_test_array)
        test_cluster_labels = cluster_model_test.predict(X_test_array)
    else:
        test_cluster_labels = cluster_model_test.fit_predict(X_test_array)

    

    X_train_output = X_train.copy()
    X_train_output['GROUP'] = train_cluster_labels 
    y_train_labeled = y_train.copy()

    df_train_clustered_output = pd.concat([X_train_output, y_train_labeled], axis=1)

    output_train_path = os.path.join(output_dir, f'{base_filename}_{cluster_name}_train_inverted.csv')
    df_train_clustered_output.to_csv(output_train_path, sep=';', decimal=',', index=False)
    print(f"CSV clusterizado (TRAIN) salvo: {output_train_path}")



    X_test_output = X_test.copy()
    X_test_output['GROUP'] = test_cluster_labels 
    y_test_labeled = y_test.copy() 
    

    df_test_clustered_output = pd.concat([X_test_output, y_test_labeled], axis=1)

    output_test_path = os.path.join(output_dir, f'{base_filename}_{cluster_name}_test_inverted.csv')
    df_test_clustered_output.to_csv(output_test_path, sep=';', decimal=',', index=False)
    print(f"CSV clusterizado (TEST) salvo: {output_test_path}")
print("\nProcessamento e salvamento de todos os CSVs concluído.")