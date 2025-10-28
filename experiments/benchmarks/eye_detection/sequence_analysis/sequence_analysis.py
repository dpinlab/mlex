from os.path import join, abspath
import sys
sys.path.append(abspath(join(__file__ , "..", "..","..","..","..")))

import matplotlib.pyplot as plt
from mlex import ensure_directory_exists
from mlex import SequenceSpanAnalyzer
from mlex import create_summary_table
from mlex import SequenceAnalyzerPlotter
from mlex import DataReader
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

def main():
    data_path = r'/data/eeg_eyestate/EEG_Eye_State.csv'

    sequence_lengths = [10, 20, 30, 40, 50]
    sequences_compositions = ['Baseline', 'kmeans', 'gmm', 'agglomerative']

    cluster_algorithms = {
        'kmeans': KMeans(n_clusters=3, n_init="auto", random_state=42),
        'gmm': GaussianMixture(n_components=3, random_state=42),
        'agglomerative': AgglomerativeClustering(n_clusters=3)
    }
    sequence_column_dict = {
        'Baseline': None,
        'kmeans': 'cluster',
        'gmm' : 'cluster',
        'agglomerative' : 'cluster'
    }
    target_column = 'eyeDetection'
    filter_data = {}

    output_dir = join(abspath(join(__file__, "..")), "sequence_analysis_results")
    ensure_directory_exists(output_dir)

    reader = DataReader(data_path, target_columns=[target_column])
    



    # df['DATA_LANCAMENTO2'] = (df['DATA_LANCAMENTO'].astype(int) // 10**9 // 86400)
    # df['DATA_LANCAMENTO3'] = df['DATA_LANCAMENTO2'] - (df['DATA_LANCAMENTO2'][0]-1)

    for composition in sequences_compositions:
        df = reader.read_df()

        df['Timestamp'] = range(1, len(df)+1)


        #df = df.drop([10386,11509,898]) 
        df2 = df.copy()
        X_full,y_full = reader.get_X_y()
        X_numeric = X_full.drop(columns=['Timestamp'])

        seq_col = sequence_column_dict[composition]
        if composition != 'Baseline':
            if composition == 'gmm':
                cluster_algorithms[composition].fit(X_numeric)
                cluster_labels = cluster_algorithms[composition].predict(X_numeric)
            else:
                cluster_labels = cluster_algorithms[composition].fit_predict(X_numeric)
            
            df2['cluster'] = cluster_labels
            df2['GROUP'] =  df2['cluster'].fillna('Unknown')
        else:
            df2['GROUP'] = 'Unknown'
        sequence_column_dict[composition] = df2



    ### Example of a custom time difference function (if needed) ###
    # def hours_diff(end, start):
    #      return (end - start).total_seconds() / 3600

    # analyzer = SequenceSpanAnalyzer(
    #     sequence_lengths=[10, 20],
    #     sequence_compositions=your_dict,
    #     group_column='GROUP',
    #     time_column='DATETIME',
    #     time_diff_fn=hours_diff,
    #     summary_bins=[
    #         ('span_0h', lambda x: x == 0),
    #         ('span_1h', lambda x: x == 1),
    #         ('span_1_24h', lambda x: 1 <= x <= 24),
    #         # etc.
    #     ]
    # )

    analyzer = SequenceSpanAnalyzer(
        sequence_lengths=sequence_lengths,
        group_column='GROUP',
        time_column='Timestamp',
       #time_diff_fn= hours_diff
    )
    analyzer.fit(sequence_column_dict)
    results = analyzer.results_

    summary_df = create_summary_table(results)
    summary_path = join(output_dir, "sequence_days_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    plot_path = join(output_dir, "sequence_days_plots.png")
    plotter = SequenceAnalyzerPlotter(
        results,
        language='en-us',
        fontsize=12,
        use_latex_font=False
    )

    plotter.plot_mean_span()
    plt.tight_layout()
    plt.savefig(join(output_dir, "mean_span.png"), dpi=600)

    plotter.plot_num_sequences()
    plt.tight_layout()
    plt.savefig(join(output_dir, "num_sequences.png"), dpi=600)

    plotter.plot_cdf()
    plt.tight_layout()
    plt.savefig(join(output_dir, "cdf.png"), dpi=600)

    plotter.plot_all(save_path=plot_path)


if __name__ == "__main__":
    main()