import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import pandas as pd
import numpy as np
import arff
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


reader = DataReader(output_base_path, target_columns=[target_column] )
X,y = reader.get_X_y()


#### split #####
splitter_tt = PastFutureSplit(proportion=0.75, timestamp_column=timestamp_column)
splitter_tt.fit(X, y)

X_train, y_train, X_test, y_test = splitter_tt.transform(X, y)

###################



df_train = pd.concat([X_train, y_train], axis=1)

output_train_path = os.path.join(output_dir, f'{base_filename}_temporal_train_inverted.csv')
df_train.to_csv(output_train_path, sep=';', decimal=',', index=False)
print(f"CSV (TRAIN) salvo: {output_train_path}")



df_test = pd.concat([X_test, y_test], axis=1)

output_test_path = os.path.join(output_dir, f'{base_filename}_temporal_test_inverted.csv')
df_test.to_csv(output_test_path, sep=';', decimal=',', index=False)
print(f"CSV (TEST) salvo: {output_test_path}")
print("\nProcessamento e salvamento de todos os CSVs concluído.")