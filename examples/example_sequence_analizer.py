from os.path import join, abspath
import sys
sys.path.append(abspath(join(__file__ , "..", "..")))

import matplotlib.pyplot as plt
from mlex import ensure_directory_exists
from mlex import SequenceSpanAnalyzer
from mlex import create_summary_table
from mlex import SequenceAnalyzerPlotter
from mlex import DataReader


def main():
    data_path = r'/data/pcpe/pcpe_04.csv'
    sequence_lengths = [i for i in range(10, 101, 10)]
    sequences_compositions = ['baseline', 'account', 'individual']
    sequence_column_dict = {
        'baseline': None,
        'account': 'CONTA_TITULAR',
        'individual': 'CPF_CNPJ_TITULAR'
    }
    target_column = 'I-d'
    filter_data = {'NATUREZA_LANCAMENTO': 'C'}

    output_dir = join(abspath(join(__file__, "..")), "04_new_sequence_days_analysis")
    ensure_directory_exists(output_dir)

    reader = DataReader(data_path, target_columns=[target_column], filter_dict=filter_data)
    df = reader.read_df()
    df['DATA_LANCAMENTO2'] = (df['DATA_LANCAMENTO'].astype(int) // 10**9 // 86400)
    df['DATA_LANCAMENTO3'] = df['DATA_LANCAMENTO2'] - (df['DATA_LANCAMENTO2'][0]-1)

    for composition in sequences_compositions:
        df2 = df.copy()
        seq_col = sequence_column_dict[composition]
        if composition != 'baseline':
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

    plot_path = join(output_dir, "sequence_days_plots.png")
    plotter = SequenceAnalyzerPlotter(results)

    plotter.plot_mean_span()
    plt.savefig(join(output_dir, "mean_span.png"))

    plotter.plot_all(save_path=plot_path)


if __name__ == "__main__":
    main()