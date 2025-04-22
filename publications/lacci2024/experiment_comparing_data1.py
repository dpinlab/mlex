import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")
sys.path.append("../../..")

from mlex import PastFutureSplit


from mlex import (
    SimplePipeline,
    SimpleRNNModel,
    SimpleLSTMModel,
    SimpleGruModel,
)

from mlex import (
    SequenceTransformer
)

from utils import equal_features

import tensorflow as tf

import keras

from itertools import cycle

from plots import Plotter

from sklearn import metrics
import cycler

from mlex.utils.analysis import LacciAnalysis

import scipy.stats as st

import pandas as pd


path_train = "/data/pcpe/pcpe_01.csv"
df_train = pd.read_csv(path_train, delimiter=';')

#df_descriptive = lacci_analysis.get_results_descriptive()
#print(df_descriptive.to_latex())


path_test = "/data/pcpe/pcpe_01.csv"
df_test =  pd.read_csv(path_test, delimiter=';')

split = PastFutureSplit()

#df_train, df_test = equal_features.match_datasets(df_train, df_test)

print(f"df train: {df_train.shape}")
print(f"df test: {df_test.shape}")


lacci_analysis_train = LacciAnalysis(df_train) 
lacci_analysis_test = LacciAnalysis(df_test) 

X, y = lacci_analysis_train.get_X_y()
#X_test, y_test = lacci_analysis_test.get_X_y()

#X_test = np.delete(X_test, 44, axis=1)

X_train, X_test, y_train, y_test = split.train_test_split(X, y)
sequence_lengths = [30, 40]
for sequence_length in sequence_lengths:
    print(f"\nRodando experimento com sequence_length = {sequence_length}")
    sequence = SequenceTransformer(sequence_length=sequence_length)
    data_test = sequence.transform(X_test, y_test)

    data_train_full = sequence.transform(X_train, y_train)

    N = len(data_train_full)
    train_size = int(0.7 * N)
    val_size = int(0.3 * N)

    data_train = data_train_full.take(train_size)
    data_val = data_train_full.skip(train_size)

    models = []
    if sequence_length == 40:
        model_lstm = SimpleLSTMModel(X_train.shape)
        models.append(model_lstm)
    else:
        model_gru = SimpleGruModel(X_train.shape)
        models.append(model_gru)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    pipelines = []
    # models = [model]

    for m in models:
        pipeline = SimplePipeline(final_model=m.get_model())
        history = m.fit(data_train, epochs=30, validation_data=data_val, callbacks = [callback])
        pipelines.append(m)

    def get_y_pred_actual(y_pred_score, y_test, sequence_length):
        y_pred = y_pred_score > np.quantile(y_pred_score, 0.95)
        y_true = y_test[:-sequence_length+1]
        return y_true, y_pred
        
    for p in pipelines:
        y_pred_score = p.predict(data_test)
        y_true, y_pred = get_y_pred_actual(y_pred_score, y_test, sequence_length)
        print("Y:")
        print(y_true)

        print("Y_pred_score:")
        print(y_pred_score) 

        print ("Y_pred:")
        print(y_pred)

        #.ravel() pra deixar unidimensional
        y_true = y_true.ravel()
        y_pred_score = y_pred_score.ravel()
        y_pred = y_pred.ravel()

        df_resultados = pd.DataFrame({
            "y_true": y_true,
            "y_pred_score": y_pred_score,
            "y_pred" : y_pred
        })

        #df_resultados.to_csv(f"saida_resultados.csv", index=False)

        if sequence_length == 40:
            df_resultados.to_csv(f"lstm_train_1_0.csv", index=False)
        else : 
            df_resultados.to_csv(f"gru_train_1_0.csv", index=False)

