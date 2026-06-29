import sys,os
from os.path import exists, join, abspath

sys.path.append(abspath(join(__file__ , "..", "..", "..")))
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from mlex.evaluation.plotter import EvaluationPlotter


# datasets = ["ai4i", "ozone", "pcpe"]
datasets=["ozone"]

sequence_lengths = [10, 20, 30, 40, 50]
num_layers = 1
hidden_size = 10
iterations = 10


# Mapeamento exato das chaves e os labels bonitos para a legenda do gráfico
model_mapping = {
    'pure-rnn': 'RNN',
    'pure-lstm': 'LSTM',
    'pure-gru': 'GRU',
    'hybrid': 'Híbrido (RNN+LSTM+GRU -> SVM)'
}

for dataset_name in datasets:
    print(f"\n==================================================")
    print(f" PROCESSANDO DATASET: {dataset_name.upper()} ")
    print(f"==================================================")
    
    # Define o caminho do arquivo baseado na pasta do dataset
    output_parquet = f"{dataset_name}-all_experiments_results.parquet"
    
    # Se o script estiver rodando da pasta pai e os parquets dentro de pastas dedicadas:
    if not exists(output_parquet) and exists(join(dataset_name, output_parquet)):
        output_parquet = join(dataset_name, output_parquet)
        
    if not exists(output_parquet):
        print(f" Atenção: Arquivo {output_parquet} não encontrado. Pulando...")
        continue

    # Caminho de salvamento dinâmico para cada base
    save_path = join(dataset_name, "results", f"{num_layers}-layer")
    os.makedirs(save_path, exist_ok=True)

    plotter = EvaluationPlotter(output_parquet)

    # Vamos gerar um gráfico comparativo contendo todas as arquiteturas por Sequence Length
    for s_len in sequence_lengths:
        print(f"Gerando gráficos para Sequence Length: {s_len}...")
        
        model_groups = []
        labels = []
        
        # Monta os grupos de iterações para cada modelo disponível
        for model_key, label_name in model_mapping.items():
            group_ids = [
                f"{dataset_name}-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{s_len}_Iteration-{i+1}"
                for i in range(iterations)
            ]
            model_groups.append(group_ids)
            labels.append(label_name)

        title_suffix = f"(Sequence Length: {s_len})"

        # 1. Plot ROC com Intervalo de Confiança (Comparando os 4 modelos juntos)
        fig, ax = plt.subplots(figsize=(8, 6))
        plotter.plot_roc_curve_with_ci(model_groups, ax=ax, labels=labels, shade=True)
        ax.set_title(f"ROC Curve Comparison - {title_suffix}")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(join(save_path, f"ROC_Comparison_S{s_len}.pdf"), format="pdf", dpi=300)
        plt.close(fig)

        # 2. Plot F1-Score History com Intervalo de Confiança (Comparando os 4 modelos juntos)
        fig, ax = plt.subplots(figsize=(8, 6))
        plotter.plot_metric_history_with_ci(model_groups, metric="f1", ax=ax)
        ax.set_title(f"F1-Score History Comparison - {title_suffix}")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(join(save_path, f"F1_Comparison_S{s_len}.pdf"), format="pdf", dpi=300)
        plt.close(fig)

    print(f"\nTodos os plots salvos com sucesso em: {save_path}")