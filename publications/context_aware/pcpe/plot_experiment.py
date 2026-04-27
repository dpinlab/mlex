import os
import sys
from os.path import join
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from mlex.evaluation.plotter import EvaluationPlotter
from mlex import DataReader, RNN, LSTM, GRU, BILSTM


model_classes = [RNN, LSTM, GRU, BILSTM]
output_parquet = "evaluation_results_full.parquet"
sequence_lengths = [10, 20, 30, 40, 50]
num_layers = 1
hidden_size = 10
iterations = 10
threshold_strategy = 'f1max'
sequences_compositions = ['baseline', 'account', 'individual']

save_path = join("results", f"{num_layers}-layer")
os.makedirs(save_path, exist_ok=True)

plotter = EvaluationPlotter(output_parquet)

def get_label(group_id: str):
    label_mapping = {
        'baseline': 'Baseline (Temporal)',
        'individual': 'Individual Context (CPF)',
        'account': 'Account Context',
    }
    for key, label in label_mapping.items():
        if f"_{key}_" in group_id:
            return label
    return "Unknown Context"

for ModelClass in model_classes:
    m_name = ModelClass.__name__
    for s_len in sequence_lengths:
        
        model_groups = []
        for comp_name in sequences_compositions:
            group_ids = [
                f"{m_name}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{s_len}_{comp_name}_{threshold_strategy}_Iteration-{i+1}"
                for i in range(iterations)
            ]
            model_groups.append(group_ids)

        model_groups = [g for g in model_groups if g]
        if not model_groups: continue

        title_suffix = f"{m_name} (Seq: {s_len})"
        labels = [get_label(g[0]) for g in model_groups]

        # Plot ROC com Intervalo de Confiança
        fig, ax = plt.subplots(figsize=(8, 6))
        # plotter.plot_roc_curve_with_ci(model_groups, ax=ax, labels=labels, shade=True)
        plotter.plot_roc_curve_with_ci(model_groups, ax=ax,labels=labels, shade=True)
        ax.set_title(f"ROC Curve - {title_suffix}")
        plt.savefig(join(save_path, f"ROC_{m_name}_S{s_len}.pdf"), format="pdf", dpi=300)
        plt.close(fig)

        # Plot F1-Score History
        fig, ax = plt.subplots(figsize=(8, 6))
        # plotter.plot_metric_history_with_ci(model_groups, metric="f1", ax=ax, labels=labels)
        plotter.plot_metric_history_with_ci(model_groups, metric="f1", ax=ax)
        ax.set_title(f"F1-Score - {title_suffix}")
        plt.savefig(join(save_path, f"F1_{m_name}_S{s_len}.pdf"), format="pdf", dpi=300)
        plt.close(fig)

print(f"Plots salvos em: {save_path}")