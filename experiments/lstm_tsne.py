import sys
from os.path import join, abspath
from os import makedirs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
import scipy.stats as stats


sys.path.append(abspath(join(__file__ , "..", "..")))

from mlex import DataReader, LSTM
from pcpe_utils import get_pcpe_dtype_dict, pcpe_preprocessing_read_func

def main():
    print("Setting up data and model...")
    path_train = r'/data/pcpe/pcpe_03.csv'
    path_test = r'/data/pcpe/pcpe_04.csv'
    target_column = 'I-d'
    filter_data = {'NATUREZA_LANCAMENTO': 'C'}
    random_state = 42
    total_runs = 30
    seq_lens = [10, 20, 30, 40, 50]
    compositions = ['baseline', 'account', 'individual']
    sequence_column_dict = {'baseline': None, 'account': 'CONTA_TITULAR', 'individual': 'CPF_CNPJ_TITULAR'}
    column_to_stratify = 'CPF_CNPJ_TITULAR'

    all_runs_data = []

    print("Loading training data...")
    reader_train = DataReader(path_train, target_columns=[target_column], dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
    X_train, y_train = reader_train.get_X_y()
    
    print("Loading test data...")
    reader_test = DataReader(path_test, target_columns=[target_column], dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
    X_test, y_test = reader_test.get_X_y()

    for sequence_composition in compositions:
        sequence_column = sequence_column_dict[sequence_composition]
        for seq_len in seq_lens:
            config_runs_data = [] # Data for this specific configuration
            scores_silhouette = []
            
            # Define output directory for this configuration
            # ./tsne_results/MODEL_NAME/SEQ_LEN/CONTEXT_COMPOSITION
            output_dir = join('RESULTS_TSNE', 'LSTM', str(seq_len), str(sequence_composition))
            makedirs(output_dir, exist_ok=True)
            
            for exec_ in range(1, total_runs+1):
                print(f"Initializing LSTM model (Seq: {seq_len}, Comp: {sequence_composition}, Run: {exec_})...")
                model_LSTM = LSTM(
                    target_column='I-d',
                    numeric_features=['DIA_LANCAMENTO','MES_LANCAMENTO','VALOR_TRANSACAO','VALOR_SALDO'],
                    categorical_features=['TIPO', 'NATUREZA_SALDO'],
                    split_stratify_column=column_to_stratify,
                    val_split=0.3,
                    context_column=sequence_column,
                    timestamp_column='DATA_LANCAMENTO',
                    filter_dict=filter_data,
                    collect_activations=True,
                    hidden_size=10,
                    seq_length=seq_len,
                    # epochs=2,
                )

                print("Training model...")
                model_LSTM.fit(X_train, y_train)

                print("Running predictions on test set...")
                
                _ = model_LSTM.predict_proba(X_test)

                print("Extracting activations...")

                predict_activations = model_LSTM.final_model.activations['predict']

                last_hiddens = [batch['last_hidden'] for batch in predict_activations]
                all_last_hiddens = np.concatenate(last_hiddens, axis=0)
                
                model_outputs = [batch['output'] for batch in predict_activations]
                all_outputs = np.concatenate(model_outputs, axis=0).flatten()
                
                print(f"Collected {len(all_last_hiddens)} samples.")
                
                # Realign with Ground Truth Labels
                activation_indices = model_LSTM.activation_indices_
                y_true_aligned = y_test.loc[activation_indices].values.flatten()
                
                print("\nClass Distribution:")
                unique, counts = np.unique(y_true_aligned, return_counts=True)
                for label, count in zip(unique, counts):
                    proportion = count / len(y_true_aligned)
                    print(f"Class {int(label)}: {count} samples ({proportion:.2%})")

                print("Running t-SNE on Last Hidden States...")

                # , random_state=random_state+exec_
                tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
                tsne_results = tsne.fit_transform(all_last_hiddens)
                score = silhouette_score(tsne_results, y_true_aligned)

                print(f"t-SNE Run {exec_}/{total_runs} - Silhouette Score: {score:.4f}")
                scores_silhouette.append(score)

                # Plotting
                print("Plotting results...")
                plt.figure(figsize=(16, 12))
                
                # Main super title
                plt.suptitle(f'LSTM Analysis - Context: {sequence_composition} - Seq_len: {seq_len}', fontsize=16)

                # Plot t-SNE
                plt.subplot(2, 2, 1)
                scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_true_aligned, cmap='coolwarm', alpha=0.6)

                handles, _ = scatter.legend_elements()
                legend_labels = ['Class 0 (True)', 'Class 1 (True)']
                plt.legend(handles, legend_labels, title="Ground Truth Class")

                plt.title('t-SNE of LSTM Last Hidden State\n(Colored by Ground Truth Labels)')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.grid(True, linestyle='--', alpha=0.6)

                # Plot Distribution of Outputs (All)
                plt.subplot(2, 2, 2)
                sns.histplot(all_outputs, bins=50, kde=True)
                plt.title('Distribution of Model Outputs (All Classes)')
                plt.xlabel('Sigmoid Output')
                plt.grid(True, linestyle='--', alpha=0.6)

                # Plot Distribution of Outputs (Class 0 - Blue)
                plt.subplot(2, 2, 3)
                sns.histplot(all_outputs[y_true_aligned == 0], bins=50, kde=True, color='blue')
                plt.title('Distribution of Model Outputs (Class 0)')
                plt.xlabel('Sigmoid Output')
                plt.grid(True, linestyle='--', alpha=0.6)

                # Plot Distribution of Outputs (Class 1 - Red)
                plt.subplot(2, 2, 4)
                sns.histplot(all_outputs[y_true_aligned == 1], bins=50, kde=True, color='red')
                plt.title('Distribution of Model Outputs (Class 1)')
                plt.xlabel('Sigmoid Output')
                plt.grid(True, linestyle='--', alpha=0.6)

                # Save plots to the structured directory
                output_file_base = join(output_dir, f'tsne_analysis_run_{exec_}')
                plt.subplots_adjust(top=0.95)
                plt.tight_layout()
                plt.savefig(f"{output_file_base}.png", dpi=300)
                plt.savefig(f"{output_file_base}.pdf")
                print(f"Plots saved to {output_file_base}.png/.pdf")
                plt.close() # Close figure to free memory

                print("\nHidden State Statistics:")
                print(f"Mean: {np.mean(all_last_hiddens):.4f}")
                print(f"Std: {np.std(all_last_hiddens):.4f}")
                print(f"Min: {np.min(all_last_hiddens):.4f}")
                print(f"Max: {np.max(all_last_hiddens):.4f}")

                # Collect data for this run
                print("Collecting data for Parquet storage...")
                run_data = pd.DataFrame({
                    'tsne_1': tsne_results[:, 0],
                    'tsne_2': tsne_results[:, 1],
                    'sigmoid_output': all_outputs,
                    'y_true': y_true_aligned,
                    'hidden_state': list(all_last_hiddens),
                    # Constant columns (broadcasted by pandas)
                    'run_id': exec_,
                    'model_name': 'LSTM',
                    'seq_len': seq_len,
                    'sequence_composition': sequence_composition,
                    'silhouette_score': score
                })
                
                config_runs_data.append(run_data)
                all_runs_data.append(run_data)
        
            # Save configuration-specific Parquet
            print(f"Saving Parquet for Seq: {seq_len}, Comp: {sequence_composition}...")
            if config_runs_data:
                config_df = pd.concat(config_runs_data, ignore_index=True)
                parquet_config_path = join(output_dir, 'experiment_results.parquet')
                config_df.to_parquet(parquet_config_path, index=False)
                print(f"Configuration results saved to {parquet_config_path}")

            print("\n\nFinal Summary of Silhouette Scores:")
            print(f"Silhouette Scores from all runs: {scores_silhouette}")
            
            avg_score = np.mean(scores_silhouette)
            std_score = np.std(scores_silhouette, ddof=1) # Use sample std dev
            median_score = np.median(scores_silhouette)
            
            # Calculate 95% Confidence Interval
            confidence = 0.95
            n = len(scores_silhouette)
            sem = stats.sem(scores_silhouette)
            ci_interval = stats.t.interval(confidence, n-1, loc=avg_score, scale=sem)
            ci_margin = ci_interval[1] - avg_score
            
            print(f"\nAverage Silhouette Score over all runs: {avg_score:.4f} ± {std_score:.4f}")
            print(f"Median Silhouette Score over all runs: {median_score:.4f}")
            print(f"95% Confidence Interval: [{ci_interval[0]:.4f}, {ci_interval[1]:.4f}] (Margin: {ci_margin:.4f})")

            # Save metrics to CSV
            metrics_data = {
                'model_name': 'LSTM',
                'seq_len': seq_len,
                'sequence_composition': sequence_composition,
                'run_ids': [list(range(1, total_runs+1))],
                'raw_scores': [scores_silhouette],
                'mean_score': avg_score,
                'median_score': median_score,
                'std_score': std_score,
                'ci_lower_95': ci_interval[0],
                'ci_upper_95': ci_interval[1],
                'ci_margin_95': ci_margin
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_csv_path = join(output_dir, 'silhouette_metrics.csv')
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"Metrics saved to {metrics_csv_path}")
            print("#" * 50)
            print("#" * 50)
            print()

    # Save to Parquet (Global)
    print("Saving aggregated results to Parquet...")
    if all_runs_data:
        final_df = pd.concat(all_runs_data, ignore_index=True)
        parquet_path = join('tsne-results', 'lstm_experiment_results_all.parquet')
        final_df.to_parquet(parquet_path, index=False)
        print(f"Global results saved to {parquet_path}")
    print("#" * 50)
    print("#" * 50)
    print("#" * 50)
    print("#" * 50)
    print()

if __name__ == "__main__":
    main()
