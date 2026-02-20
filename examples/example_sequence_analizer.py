from os.path import join, abspath
import sys
sys.path.append(abspath(join(__file__ , "..", "..")))

import matplotlib.pyplot as plt
from mlex import ensure_directory_exists
from mlex import SequenceSpanAnalyzer
from mlex import create_summary_table
from mlex import SequenceAnalyzerPlotter
from mlex import DataReader
from pcpe_utils import get_pcpe_dtype_dict, pcpe_preprocessing_read_func


def main():
    data_path = r'/data/pcpe/pcpe_04.csv'
    sequence_lengths = [i for i in range(10, 51, 10)]
    sequences_compositions = ['Baseline', 'Individual', 'Account']
    sequence_column_dict = {
        'Baseline': None,
        'Individual': 'CPF_CNPJ_TITULAR',
        'Account': 'CONTA_TITULAR',
    }
    target_column = 'I-d'
    filter_data = {'NATUREZA_LANCAMENTO': 'C'}

    output_dir = join(abspath(join(__file__, "..")), "04_sequence_days_analysis")
    ensure_directory_exists(output_dir)

    reader = DataReader(data_path, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
    df = reader.read_df()
    df['DATA_LANCAMENTO2'] = (df['DATA_LANCAMENTO'].astype(int) // 10**9 // 86400)
    df['DATA_LANCAMENTO3'] = df['DATA_LANCAMENTO2'] - (df['DATA_LANCAMENTO2'][0]-1)

    for composition in sequences_compositions:
        df2 = df.copy()
        seq_col = sequence_column_dict[composition]
        if composition != 'Baseline':
            df2['GROUP'] = df2[seq_col].fillna('Unknown')
        else:
            df2['GROUP'] = 'Unknown'
        sequence_column_dict[composition] = df2



    ### Example of a custom time difference function (if needed) ###
    # def hours_diff(end, start):
    #     return (end - start).total_seconds() / 3600

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
        time_column='DATA_LANCAMENTO3'
    )
    analyzer.fit(sequence_column_dict)
    results = analyzer.results_

    summary_df = create_summary_table(results)
    summary_path = join(output_dir, "sequence_days_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    plot_path = join(output_dir, "sequence_days_plots.pdf")
    plotter = SequenceAnalyzerPlotter(
        results,
        language='en-us',
        fontsize=12,
        use_latex_font=True
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