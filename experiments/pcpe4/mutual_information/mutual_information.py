import pandas as pd
import numpy as np
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from sklearn.feature_selection import mutual_info_classif 
import matplotlib.pyplot as plt
import seaborn as sns

models = ['RNN', 'LSTM', 'GRU','BILSTM']
for model in models:

    output_dir = f'/data/isa/mlex/experiments/pcpe4/{model}/results_ci_withcontext'
    CONSOLIDATED_FILE_NAME = 'evaluation.parquet'
    comparison_pairs = [
        ('temporal', 'Feature_individual'),
        ('temporal', 'Feature_account'),
        ('Feature_individual', 'Feature_account')
    ]
    BINARIZATION_THRESHOLD = 0.5 
    METADATA_COLUMN = 'model_id' 
    OUTPUT_IMAGE = f'mutual_information_plot_{model}.png'
    OUTPUT_IMAGE_HEATMAP = f'mutual_information_heatmap_{model}.png'
    OUTPUT_TABLE_CSV = f'mutual_information_table_{model}.csv'

    def extract_params_from_name(name: str) -> tuple[int, str, int]:
        """Extrai sequence_length, composition e iteration do nome do experimento."""
        seq_match = re.search(r'SequenceLength-(\d+)', name)
        seq_length = int(seq_match.group(1)) if seq_match else None
        
        comp_match = re.search(r'SequenceLength-\d+_([a-zA-Z0-9_]+)_f1max', name)
        composition = comp_match.group(1) if comp_match else None
        
        iter_match = re.search(r'Iteration-(\d+)', name)
        iteration = int(iter_match.group(1)) if iter_match else None
        
        return seq_length, composition, iteration


    file_path = os.path.join(output_dir, CONSOLIDATED_FILE_NAME)
    print(f"Carregando arquivo consolidado (vetores de instância por linha): {file_path}")

    try:
        df_raw = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Erro ao carregar o arquivo consolidado: {e}")
        sys.exit()

    df_list = []

    print("Reconstruindo DataFrame: Desempacotando vetores y_true e y_pred...")

    global_instance_ids = None

    for index, row in df_raw.iterrows():
        
        experiment_name = row.get(METADATA_COLUMN)
        if not experiment_name:
            continue

        seq_len, comp, iteration = extract_params_from_name(experiment_name)
        
        if seq_len is None or comp is None or iteration is None:
            continue
        
        try:
            y_true_vector = np.array(row['y_true']).astype(int)
            y_pred_vector = np.array(row['y_pred']).astype(int)

        except Exception as e:
            print(f"⚠️ Erro ao converter vetores binários na linha {index}: {e}. Pulando.")
            continue

        current_instance_ids = np.arange(len(y_true_vector))
        df_instance = pd.DataFrame({
            'y_true': y_true_vector,
            'y_pred': y_pred_vector,
            'sequence_index': current_instance_ids, 
            'sequence_length': int(seq_len),
            'composition': comp,
            'iteration': int(iteration)
        })
        
        df_list.append(df_instance)

    if not df_list:
        print("❌ Não foi possível reconstruir o DataFrame de instâncias. Encerrando.")
        sys.exit()
        
    df_long = pd.concat(df_list, ignore_index=True)
    print(f"DataFrame reconstruído com {len(df_long)} predições de instância.")



    print("Agregando 10 iterações em média (voto majoritário) por sequência/configuração...")

    df_aggregated = df_long.groupby(['sequence_index', 'sequence_length', 'composition'])['y_pred'].mean().reset_index()

    df_aggregated.rename(columns={'y_pred': 'y_pred_avg'}, inplace=True)

    df_aggregated['y_pred_bin_avg'] = (df_aggregated['y_pred_avg'] > BINARIZATION_THRESHOLD).astype(int)

    df_true_subset = df_long[['sequence_index', 'sequence_length', 'composition', 'y_true']].drop_duplicates()
    df_true_subset = df_true_subset.drop_duplicates(subset=['sequence_index', 'sequence_length', 'composition'])


    df_aggregated = df_aggregated.merge(
        df_true_subset, 
        on=['sequence_index', 'sequence_length', 'composition'], 
        how='left'
    )



    print("\n=== Cálculo de Mutual Information (MI) nas Decisões Consolidadas ===")

    mi_results = []
    processed_seq_lens = sorted(df_aggregated['sequence_length'].unique())

    for seq_len in processed_seq_lens:
        print(f"\n[Sequence Length: {seq_len}]")
        
        df_pivot = df_aggregated[df_aggregated['sequence_length'] == seq_len].pivot(
            index='sequence_index',
            columns='composition',
            values='y_pred_bin_avg'
        ).fillna(-1)
        
        for comp_A, comp_B in comparison_pairs:
            
            if comp_A in df_pivot.columns and comp_B in df_pivot.columns:
                
                X = df_pivot[[comp_A]].values.astype(int) 
                y = df_pivot[comp_B].values.astype(int)  
                
                mi_value = mutual_info_classif(X, y, random_state=42)[0]
                
                print(f"  MI ({comp_A} vs {comp_B}): {mi_value:.4f}")

                mi_results.append({
                    'sequence_length': seq_len,
                    'model_A': comp_A,
                    'model_B': comp_B,
                    'mutual_information': mi_value
                })


    df_mi = pd.DataFrame(mi_results)

    if not df_mi.empty:
        print("\n\n=== Resumo Final do Mutual Information (Decisão Consolidada) ===")
        print(df_mi)
        
        df_mi.to_csv(f'analysis_mutual_information_summary_{model}.csv', index=False)
        print("\nResultados salvos em 'analysis_mutual_information_summary.csv'")
    else:
        print("\nNão foi possível calcular o Mutual Information.")

    df_mi['Comparison'] = df_mi['model_A'] + ' vs ' + df_mi['model_B']


    print(f"Gerando Line Plot -> {OUTPUT_IMAGE}")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_mi, x='sequence_length', y='mutual_information',
        hue='Comparison', style='Comparison', markers=True, dashes=False, linewidth=2.5, markersize=8, palette='viridis'
    )
    plt.title('Mutual Information between contexts(MI)', pad=20)
    plt.xticks(sorted(df_mi['sequence_length'].unique()))
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    plt.close() 



    print(f"Gerando Heatmap -> {OUTPUT_IMAGE_HEATMAP}")

    heatmap_data = df_mi.pivot(index='Comparison', columns='sequence_length', values='mutual_information')

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heatmap_data,
        annot=True,       
        fmt=".4f",        
        cmap="YlGnBu",    
        linewidths=.5,     
        cbar_kws={'label': 'Mutual Information (bits)'}
    )
    plt.title('Matriz de Informação Mútua por Tamanho de Sequência', pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_HEATMAP)
    plt.show()

   
    print(f"Salvando Tabela Matriz -> {OUTPUT_TABLE_CSV}")
    heatmap_data.to_csv(OUTPUT_TABLE_CSV)
    print("\nTabela gerada:")
    print(heatmap_data)