import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from mlex.analysis.sequence_span_analyzer import create_summary_table
from mlex.utils.utils import get_first_middle_last_sequence_len


class SequenceAnalyzerPlotter:
    """Classe para plotar resultados do SequenceSpanAnalyzer."""

    def __init__(self, results: Dict[str, Dict[int, Dict[str, Any]]],
                 language: str = 'en-us',
                 fontsize: int = 12,
                 use_latex_font: bool = False):
        """
        Inicializa o plotter.

        Args:
            results (Dict): Dicionário com os resultados da análise.
            language (str): Idioma para os títulos e legendas ('en-us' ou 'pt-br').
            fontsize (int): Tamanho da fonte base para todos os textos dos gráficos.
            use_latex_font (bool): Se True, usa uma engine LaTeX para renderizar o texto.
        """
        self.results = results
        self.language = language
        self.fontsize = fontsize

        # --- Configuração Centralizada de Fontes ---
        font_config = {
            'font.size': self.fontsize,
            'axes.titlesize': self.fontsize + 2,
            'axes.labelsize': self.fontsize,
            'xtick.labelsize': self.fontsize,
            'ytick.labelsize': self.fontsize,
            'legend.fontsize': self.fontsize,
            'figure.titlesize': self.fontsize + 8
        }
        plt.rcParams.update(font_config)
        
        if use_latex_font:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
            })

        plt.style.use('seaborn-v0_8-colorblind')
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.linestyles = ['-', '--', ':']
        self._set_labels()

    def _set_labels(self):
        """Define os rótulos dos gráficos com base no idioma selecionado."""
        labels = {
            'en-us': {
                'mean_span_title': 'Mean Span by Sequence Length',
                'mean_span_xlabel': 'Sequence Length',
                'mean_span_ylabel': 'Mean Span',
                'num_sequences_title': 'Number of Sequences by Configuration',
                'num_sequences_xlabel': 'Sequence Length',
                'num_sequences_ylabel': 'Number of Sequences',
                'cdf_title': 'CDF of Spans',
                'cdf_xlabel': 'Span',
                'cdf_ylabel': 'Cumulative Probability',
                'percentage_title': 'Percentage of Sequences by Span Range',
                'percentage_xlabel': 'Span Range',
                'percentage_ylabel': 'Percentage of Sequences',
                'main_title': 'Sequence Span Analysis'
            },
            'pt-br': {
                'mean_span_title': 'Média de Intervalo por Comprimento da Sequência',
                'mean_span_xlabel': 'Comprimento da Sequência',
                'mean_span_ylabel': 'Média de Intevalo',
                'num_sequences_title': 'Número de Sequências por Configuração',
                'num_sequences_xlabel': 'Comprimento da Sequência',
                'num_sequences_ylabel': 'Número de Sequências',
                'cdf_title': 'CDF de Intervalos',
                'cdf_xlabel': 'Intevalo',
                'cdf_ylabel': 'Probabilidade Cumulativa',
                'percentage_title': 'Porcentagem de Sequências por Faixa de Intevalos',
                'percentage_xlabel': 'Faixa de Intervalos',
                'percentage_ylabel': 'Porcentagem de Sequências',
                'main_title': 'Análise de Intervalos de Sequência'
            }
        }
        if self.language not in labels:
            raise ValueError(f"Language '{self.language}' not supported. Please use 'en-us' or 'pt-br'.")
        self.current_labels = labels[self.language]

    def plot_mean_span(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        for composition in self.results.keys():
            seq_lengths = list(self.results[composition].keys())
            mean_spans = [self.results[composition][seq_len]['mean'] for seq_len in seq_lengths]
            ax.plot(seq_lengths, mean_spans, marker='o', label=composition, linewidth=2)

        ax.set_xlabel(self.current_labels['mean_span_xlabel'])
        ax.set_ylabel(self.current_labels['mean_span_ylabel'])
        ax.set_title(self.current_labels['mean_span_title'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_num_sequences(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        compositions = list(self.results.keys())
        if compositions:
            seq_lengths = list(self.results[compositions[0]].keys())
            x = np.arange(len(seq_lengths))
            width = 0.25
            max_y = 0
            for i, composition in enumerate(compositions):
                num_sequences = [self.results[composition][seq_len]['num_sequences'] for seq_len in seq_lengths]
                max_y = np.max(np.append(num_sequences, max_y))
                ax.bar(x + i * width, num_sequences, width, label=composition, alpha=0.8)

            max_y = max_y * 1.25
            ax.set_xlabel(self.current_labels['num_sequences_xlabel'])
            ax.set_ylabel(self.current_labels['num_sequences_ylabel'])
            ax.set_title(self.current_labels['num_sequences_title'])
            ax.set_ylim([0, max_y])
            ax.set_xticks(x + width / (len(compositions) -1 if len(compositions) > 1 else 2))
            ax.set_xticklabels(seq_lengths)
            ax.legend()
            ax.grid(True, alpha=0.3)
        return ax

    def plot_cdf(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
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
        ax.set_xlabel(self.current_labels['cdf_xlabel'])
        ax.set_ylabel(self.current_labels['cdf_ylabel'])
        ax.set_title(self.current_labels['cdf_title'])
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
                    ax.bar(indices - (bar_width * (n_comp - 1) / 2) + i * bar_width,
                           percentages,
                           width=bar_width,
                           label=f'{composition}-{first_seq_len}',
                           alpha=0.8)

            ax.set_xlabel(self.current_labels['percentage_xlabel'])
            ax.set_ylabel(self.current_labels['percentage_ylabel'])
            ax.set_title(self.current_labels['percentage_title'])
            ax.set_xticks(indices)
            ax.set_xticklabels(bin_pct_cols, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)
        return ax

    def plot_all(self, save_path: Optional[str] = None, show: bool = True):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(self.current_labels['main_title'], fontsize=16, fontweight='bold')

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
        plt.rcdefaults()
