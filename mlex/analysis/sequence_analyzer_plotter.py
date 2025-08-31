import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from typing import Dict, Any, Optional, List
from mlex.analysis.sequence_span_analyzer import create_summary_table
from mlex.utils.utils import get_first_middle_last_sequence_len


class SequenceAnalyzerPlotter:
    """Class for plotting results from SequenceSpanAnalyzer."""

    def __init__(self, results: Dict[str, Dict[int, Dict[str, Any]]]):
        self.results = results
        plt.style.use('seaborn-v0_8-colorblind')
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.linestyles = ['-', '--', ':']


    def plot_mean_span(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        for composition in self.results.keys():
            seq_lengths = list(self.results[composition].keys())
            mean_spans = [self.results[composition][seq_len]['mean'] for seq_len in seq_lengths]
            ax.plot(seq_lengths, mean_spans, marker='o', label=composition, linewidth=2)

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Mean Span')
        ax.set_title('Mean Span by Sequence Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax


    def plot_num_sequences(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        compositions = list(self.results.keys())
        if compositions:
            seq_lengths = list(self.results[compositions[0]].keys())
            x = np.arange(len(seq_lengths))
            width = 0.25
            max_y = 0
            for i, composition in enumerate(compositions):
                num_sequences = [self.results[composition][seq_len]['num_sequences'] for seq_len in seq_lengths]
                max_y = np.max(np.append(num_sequences, max_y))
                ax.bar(x + i*width, num_sequences, width, label=composition, alpha=0.8)

            max_y = max_y * 1.25
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('Number of Sequences')
            ax.set_title('Number of Sequences by Configuration')
            ax.set_ylim([0, max_y])
            ax.set_xticks(x + width)
            ax.set_xticklabels(seq_lengths)
            ax.legend()
            ax.grid(True, alpha=0.3)
        return ax


    def plot_cdf(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        for i, composition in enumerate(self.results.keys()):
            for j, seq_len in enumerate(get_first_middle_last_sequence_len(list(self.results[composition].keys()))):
                spans = self.results[composition][seq_len]['spans']
                if spans:
                    sorted_spans = np.sort(spans)
                    unique_x, counts = np.unique(sorted_spans, return_counts=True)
                    cdf = np.cumsum(counts) / counts.sum()
                    ax.plot(unique_x, cdf,
                             label=f'{composition}-{seq_len}',
                             color=self.colors[i], linestyle=self.linestyles[j % len(self.linestyles)])

        ax.set_xscale('log', base=10)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        xticks = ax.get_xticks()
        xticks = [tick for tick in xticks if tick > 0]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=45, ha="right")
        ax.set_xlabel('Span')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('CDF of Spans')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax


    def plot_percentage(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        summary_df = create_summary_table(self.results)
        bin_pct_cols = [col for col in summary_df.columns if col.endswith('_pct')]

        if not summary_df.empty and bin_pct_cols:
            all_compositions = summary_df['Composition'].unique()
            n_comp = len(all_compositions)
            bar_width = 0.8 / n_comp
            indices = np.arange(len(bin_pct_cols))

            for i, composition in enumerate(all_compositions):
                comp_data = summary_df[summary_df['Composition'] == composition]
                if not comp_data.empty:
                    first_seq_len = comp_data['Sequence_Length'].iloc[-1]
                    percentages = comp_data[comp_data['Sequence_Length'] == first_seq_len][bin_pct_cols].values.flatten()
                    ax.bar(indices - (bar_width * (n_comp-1) / 2) + i * bar_width,
                           percentages,
                           width=bar_width,
                           label=f'{composition}-{first_seq_len}',
                           alpha=0.8)

            ax.set_xlabel('Span Range')
            ax.set_ylabel('Percentage of Sequences')
            ax.set_title('Percentage of Sequences by Span Range')
            ax.set_xticks(indices)
            ax.set_xticklabels(bin_pct_cols, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)
        return ax


    def plot_all(self, save_path: Optional[str] = None, show: bool = True):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sequence Span Analysis', fontsize=16, fontweight='bold')

        self.plot_mean_span(axes[0, 0])
        self.plot_num_sequences(axes[0, 1])
        self.plot_cdf(axes[1, 0])
        self.plot_percentage(axes[1, 1])

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {save_path}")
        if show:
            plt.show()
        plt.close(fig)
