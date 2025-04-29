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
    SimpleBiLSTMModel,
)

from mlex import (
    SequenceTransformer
)

import tensorflow as tf

import keras

from itertools import cycle

from plots import Plotter

from sklearn import metrics
import cycler

from mlex.utils.analysis import LacciAnalysis

import scipy.stats as st

import pandas as pd


path = "/data/pcpe/pcpe_03.csv"
df = pd.read_csv(path, delimiter=';', decimal=',')


lacci_analysis = LacciAnalysis(df) 
df_descriptive = lacci_analysis.get_results_descriptive()
print(df_descriptive.to_latex())

split = PastFutureSplit()
X, y = lacci_analysis.get_X_y()
X_train, X_test, y_train, y_test = split.train_test_split(X, y)
sequence_lengths = [20, 30, 40, 50]
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
    
    model_bilstm = SimpleBiLSTMModel(X_train.shape)
    model_rnn = SimpleRNNModel(X_train.shape)
    model_lstm = SimpleLSTMModel(X_train.shape)
    model_gru = SimpleGruModel(X_train.shape)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    pipelines = []
    models = [model_bilstm, model_rnn, model_lstm, model_gru]
    #  models = [model]

    for m in models:
        pipeline = SimplePipeline(final_model=m.get_model())
        history = m.fit(data_train, epochs=30, validation_data=data_val, callbacks = [callback])
        pipelines.append(m)

    def get_y_pred_actual(y_pred_score, y_test, sequence_length):
        y_pred = y_pred_score > np.quantile(y_pred_score, 0.95)
        y_true = y_test[:-sequence_length+1]
        return y_true, y_pred
        
    list_y_pred_score = []
    list_ys_true_pred = []
    for p in pipelines:
        y_pred_score = p.predict(data_test)
        y_true, y_pred = get_y_pred_actual(y_pred_score, y_test, sequence_length)
        list_ys_true_pred.append((y_true, y_pred, y_pred_score))

    names = [ 'BILSTM','RNN', 'LSTM', 'GRU']
    name_cycler = cycle(names)

    plotter = Plotter()
    for y_true, y_pred, y_pred_score in list_ys_true_pred:
        plotter.plot_matrix(y_true, y_pred,name_cycler, filename=f"confusion_{sequence_length}_2layers_{name_cycler}")



    title = "ROC"
    fig, ax = plt.subplots()
    ax.plot([0,1], [0,1], "k--",linewidth=4, label='random classifier')
    for y_true, y_pred, y_pred_score in list_ys_true_pred:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_score)
        auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr,  linewidth=4, label=f"{next(name_cycler)} (AUC = {round(auc,2) })")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=16)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=16)
    ax.set_title(f"Receiver Operating Characteristic \n {title}", fontsize=18)
    ax.legend(loc="lower right")
    plt.savefig(f"roc_{sequence_length}_2layers_{name_cycler}.pdf")

    plt.show()

    def roc(y_true, y_pred_score):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_score)
        auc = metrics.auc(fpr, tpr)
        return auc


    def roc_samples(y_t, y_s):
        N = len(y_t)
        rocs = []
        for b in range(30):
            choices = np.random.choice(N, size=N)
            ys_t = y_t[choices]
            ys_p = y_s[choices]
            fpr, tpr, thresholds = metrics.roc_curve(ys_t, ys_p)
            auc = metrics.auc(fpr, tpr)
            rocs.append(auc)
        return rocs

    cis = []
    for y_true, y_pred, y_pred_score in list_ys_true_pred:
        rocs = roc_samples(y_true, y_pred_score)
        roc_mean = np.mean(rocs)
        roc_std = np.std(rocs)
        roc_lb = roc_mean - 1.96 * roc_std
        roc_up = roc_mean + 1.96 * roc_std
        ci = f'{round(roc_mean, 2)} {round(roc_lb, 2), round(roc_up, 2)}'
        cis.append(ci)

    print(f"2 Camadas {name_cycler}")
    print(cis)
    df_ci = pd.DataFrame({
            "Model": names,
            "95% Confidence Interval": cis
        })
    print(df_ci.to_latex(index=False, escape=True))