import sys
import os
from os.path import exists, join, abspath
from itertools import product

sys.path.append(abspath(join(__file__, "..", "..", "..")))
import matplotlib.pyplot as plt
import numpy as np
from mlex.evaluation.plotter import EvaluationPlotter

datasets = ["ai4i", "ozone", "occupancy", "eye-detection", "cobot"]

sequence_lengths = [10, 20, 30, 40, 50]
hidden_sizes = [10, 32]
num_layers_options = [1, 2]
iterations = 10

# Valores do Grid do SVM para busca
svm_C_grid = [0.1, 1.0, 10.0]
svm_gamma_grid = ["scale", "auto"]

model_mapping = {
    'pure-rnn': 'RNN',
    'pure-lstm': 'LSTM',
    'pure-gru': 'GRU'
}

for dataset_name in datasets:
    print(f"\n==================================================")
    print(f" PROCESSANDO GRÁFICOS: {dataset_name.upper()} ")
    print(f"==================================================")
    
    output_parquet = f"{dataset_name}-all_experiments_results.parquet"
    if not exists(output_parquet) and exists(join(dataset_name, output_parquet)):
        output_parquet = join(dataset_name, output_parquet)
        
    if not output_parquet or not exists(output_parquet):
        print(f" Atenção: Arquivo '{output_parquet}' não encontrado. Pulando...")
        continue

    print(f" Carregando resultados de: {output_parquet}")
    plotter = EvaluationPlotter(output_parquet)
    
    real_run_ids = set()
    real_run_ids.update(plotter.df.index.astype(str).tolist())
    for col in plotter.df.columns:
        if plotter.df[col].dtype == 'object':
            real_run_ids.update(plotter.df[col].dropna().astype(str).unique().tolist())

    for num_layers, hidden_size in product(num_layers_options, hidden_sizes):
        for s_len in sequence_lengths:
            
            model_groups = []
            labels = []
            has_pures = True
            
            # 1. Primeiro agrupamos os modelos PUROS
            for model_key, label_name in model_mapping.items():
                base_pattern = f"{dataset_name}-{model_key}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{s_len}_Iteration-"
                matched_ids = [rid for rid in real_run_ids if base_pattern in rid]
                
                if len(matched_ids) >= iterations:
                    matched_ids = sorted(matched_ids, key=lambda x: int(x.split("Iteration-")[1].split("_")[0]))[:iterations]
                    model_groups.append(matched_ids)
                    labels.append(label_name)
                else:
                    has_pures = False

            if not has_pures:
                continue  # Se não tiver as redes base prontas, pula

            # 2. Agora varremos o GRID COMPLETO do Híbrido e injetamos TODOS que existirem no mesmo gráfico
            hybrid_count = 0
            for svm_C, svm_gamma in product(svm_C_grid, svm_gamma_grid):
                hybrid_pattern = f"{dataset_name}-hybrid_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{s_len}_Iteration-"
                # O ID do híbrido precisa conter o padrão base E as configurações do SVM
                matched_ids = [
                    rid for rid in real_run_ids 
                    if hybrid_pattern in rid and f"svmC-{svm_C}_" in rid and f"svmGamma-{svm_gamma}" in rid
                ]
                
                if len(matched_ids) >= iterations:
                    matched_ids = sorted(matched_ids, key=lambda x: int(x.split("Iteration-")[1].split("_")[0]))[:iterations]
                    model_groups.append(matched_ids)
                    labels.append(f"Híbrido (C={svm_C}, γ={svm_gamma})")
                    hybrid_count += 1

            # Só gera o gráfico se tiver pelo menos uma variação do híbrido encontrada
            if hybrid_count == 0:
                continue

            save_path = join(dataset_name, "results", f"{num_layers}-layer", f"hidden-{hidden_size}")
            os.makedirs(save_path, exist_ok=True)

            title_suffix = f"(L={num_layers}, H={hidden_size}, S={s_len})"

            # 1. Plot ROC Curve com IC (Comparando puros + todas as variações encontradas do híbrido)
            fig, ax = plt.subplots(figsize=(10, 7)) # Aumentado levemente para acomodar mais legendas
            plotter.plot_roc_curve_with_ci(model_groups, ax=ax, labels=labels, shade=True)
            ax.set_title(f"ROC Curve Comparison - {dataset_name.upper()} {title_suffix}")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(join(save_path, f"ROC_Comparison_S{s_len}.pdf"), format="pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

            # 2. Plot F1-Score History com IC
            fig, ax = plt.subplots(figsize=(10, 7))
            plotter.plot_metric_history_with_ci(model_groups, metric="f1", ax=ax)
            ax.set_title(f"F1-Score History Comparison - {dataset_name.upper()} {title_suffix}")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(join(save_path, f"F1_Comparison_S{s_len}.pdf"), format="pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"   Gráficos gerados (Puros + {hybrid_count} Híbridos) para S={s_len} em {save_path}")

print("\n[FIM] Todos os plots de comparação metodológica foram gerados com sucesso.")