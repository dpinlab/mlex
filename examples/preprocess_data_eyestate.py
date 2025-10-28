import pandas as pd
import numpy as np
import arff

data_path = r'/data/eeg_eyestate/EEG_Eye_State.arff'

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


df.to_csv(r'/data/eeg_eyestate/EEG_Eye_State.csv', sep=';', decimal=',', index=False)