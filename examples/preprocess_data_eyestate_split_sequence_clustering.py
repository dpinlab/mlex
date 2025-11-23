import sys
import os
import pandas as pd
import numpy as np
import arff
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from mlex import DataReader, PreProcessingTransformer, ContextAware, PastFutureSplit
import torch
from scipy.stats import mode 


data_path = r'/data/eeg_eyestate/EEG_Eye_State.arff'
output_dir = r'/data/eeg_eyestate'
base_filename = 'EEG_Eye_State'
output_base_path = os.path.join(output_dir, f'{base_filename}.csv')
target_column = 'eyeDetection'
timestamp_column= 'Timestamp'
sequence_lengths = [10, 20, 30, 40, 50] 

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


with open(data_path, 'r') as f:
    raw = arff.load(f)
columns = [attr[0] for attr in raw['attributes']]
df = pd.DataFrame(raw['data'], columns=columns)

start_time = pd.Timestamp('2025-01-01 00:00:00.000')
interval = pd.to_timedelta('7.8ms')
df['Timestamp'] = [start_time + i * interval for i in range(len(df))]

df = df.drop([10386,11509, 898]) 
df['GROUP'] = 'Unknown'

# Salvamento do CSV base (opcional, pode ser mantido ou removido dependendo da necessidade)
df.to_csv(output_base_path, sep=';', decimal=',', index=False)
print(f"CSV base salvo: {output_base_path}")

df.drop(columns=['GROUP'], inplace=True)
reader = DataReader(output_base_path, target_columns=[target_column] )
X,y = reader.get_X_y()

#### split #####
splitter_tt = PastFutureSplit(proportion=0.75, timestamp_column=timestamp_column)
splitter_tt.fit(X, y)
X_train, y_train, X_test, y_test = splitter_tt.transform(X, y)

# Preprocessor deve ser fitado ANTES de criar janelas
numeric_cols_for_preprocessing = [col for col in X.columns.to_list() if col not in ['Timestamp','GROUP']]
preprocessor = PreProcessingTransformer(target_column=[target_column],numeric_features=numeric_cols_for_preprocessing,categorical_features=[],context_feature=['GROUP'],handle_unknown='ignore')
preprocessor.fit(X_train, y_train)

X_train_array, _ = preprocessor.transform(X_train, y_train)
X_test_array, _ = preprocessor.transform(X_test, y_test)

# --- 2. FUNÇÕES DE TRANSFORMAÇÃO DE SEQUÊNCIA ---

def create_sequential_window_data(data_array: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    n_instances, n_features = data_array.shape    
    n_windows = n_instances - seq_length + 1
    X_seq = np.zeros((n_windows, n_features * seq_length))
    
    for i in range(n_windows):
        window = data_array[i:i + seq_length, :]
        X_seq[i] = window.flatten() 
        
    start_indices = np.arange(n_windows)
    
    return X_seq, start_indices

def expand_sequential_labels(start_indices: np.ndarray, seq_labels: np.ndarray, total_instances: int, seq_length: int) -> np.ndarray:
    """
    Mapeia os rótulos de cluster de volta para cada instância original usando a moda.
    """
    all_assignments = [[] for _ in range(total_instances)]
    
    for i, start_idx in enumerate(start_indices):
        label = seq_labels[i]
        for j in range(seq_length):
            instance_idx = start_idx + j
            if instance_idx < total_instances:
                all_assignments[instance_idx].append(label)
    
    final_labels = np.zeros(total_instances, dtype=int)
    
    for i in range(total_instances):
        if all_assignments[i]:
            final_labels[i] = mode(all_assignments[i], keepdims=True).mode[0]
        else:
            final_labels[i] = -1 
    return final_labels

# --- 3. EXECUÇÃO DA CLUSTERIZAÇÃO SEQUENCIAL ---

def get_cluster_model(cluster_name):
    if cluster_name == 'kmeans':
        return KMeans(n_clusters=3, n_init="auto", random_state=42)
    elif cluster_name == 'gmm':
        return GaussianMixture(n_components=3, random_state=42)
    elif cluster_name == 'agglomerative':
        return AgglomerativeClustering(n_clusters=3)
    
def predict_agglomerative_labels(X_train_seq, train_seq_labels, X_test_seq):
    """
    Usa um classificador KNN (k=1) para mapear os dados de teste para os clusters 
    de Agglomerative aprendidos no treino, simulando o .predict().
    """
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_seq, train_seq_labels)
    test_seq_labels = knn.predict(X_test_seq)
    return test_seq_labels

cluster_names = ['kmeans', 'gmm', 'agglomerative']

for seq_len in sequence_lengths:
    print(f"\n======== Processando Sequence Length: {seq_len} ========")
    
   
    X_train_seq, train_start_indices = create_sequential_window_data(X_train_array, seq_len)
    
    X_test_seq, test_start_indices = create_sequential_window_data(X_test_array, seq_len)
    
    for cluster_name in cluster_names:
        print(f"\n--- Clusterização {cluster_name} (SeqLen={seq_len}) ---")
        
       
        cluster_model_train = get_cluster_model(cluster_name) 
        
        if cluster_name == 'gmm' or cluster_name == 'kmeans':
            cluster_model_train.fit(X_train_seq)
            train_seq_labels = cluster_model_train.predict(X_train_seq)
        else:
            cluster_model_train.fit(X_train_seq)
            train_seq_labels = cluster_model_train.labels_

        train_cluster_labels = expand_sequential_labels(
            train_start_indices, 
            train_seq_labels, 
            len(X_train), 
            seq_len
        )


        if cluster_name == 'gmm' or cluster_name == 'kmeans':   
            test_seq_labels = cluster_model_train.predict(X_test_seq)
        else:
            test_seq_labels = predict_agglomerative_labels(
                X_train_seq, train_seq_labels, X_test_seq
            )
        
        test_cluster_labels = expand_sequential_labels(
            test_start_indices, 
            test_seq_labels, 
            len(X_test), 
            seq_len
        )


       
        X_train_output = X_train.copy()
        X_train_output['GROUP'] = train_cluster_labels 
        y_train_labeled = y_train.copy()

        df_train_clustered_output = pd.concat([X_train_output, y_train_labeled], axis=1)
        output_train_path = os.path.join(output_dir, f'{base_filename}_{cluster_name}_Seq{seq_len}_train_inverted.csv')
        df_train_clustered_output.to_csv(output_train_path, sep=';', decimal=',', index=False)
        print(f"CSV clusterizado (TRAIN) salvo: {output_train_path}")

        X_test_output = X_test.copy()
        X_test_output['GROUP'] = test_cluster_labels 
        y_test_labeled = y_test.copy() 

        df_test_clustered_output = pd.concat([X_test_output, y_test_labeled], axis=1)
        output_test_path = os.path.join(output_dir, f'{base_filename}_{cluster_name}_Seq{seq_len}_test_inverted.csv')
        df_test_clustered_output.to_csv(output_test_path, sep=';', decimal=',', index=False)
        print(f"CSV clusterizado (TEST) salvo: {output_test_path}")

print("\nProcessamento e salvamento de todos os CSVs sequenciais concluído.")