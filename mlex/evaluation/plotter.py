import matplotlib.pyplot as plt
from os.path import join
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List, Union


class EvaluationPlotter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._load_data()

    def _load_data(self):
        self.df = pq.read_table(self.file_path).to_pandas()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

    def plot_roc_curve(self, model_ids: Union[str, List[str]] = None, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        model_ids = [model_ids] if isinstance(model_ids, str) else model_ids

        ax.plot([0, 1], [0, 1], "k--", linewidth=4, label='random classifier')
        for model_id in model_ids:
            row = self.df[self.df['model_id'] == model_id].squeeze()
            if len(row['fpr']) > 0:
                label = f"{model_id.split('_')[2].capitalize()} context (AUC = {row['auc_roc']:.2f})"
                ax.plot(row['fpr'], row['tpr'], linewidth=4, label=label)

        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=16)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=16)
        ax.set_title(f"Receiver Operating Characteristic\n{model_ids[0].split('_')[0].upper()}", fontsize=18)
        ax.legend(loc="lower right")
        return ax

    def plot_pr_curve(self, model_ids: Union[str, List[str]] = None, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        model_ids = [model_ids] if isinstance(model_ids, str) else model_ids

        y_true = self.df[self.df['model_id'].isin(model_ids)].iloc[0,].squeeze()['y_true']
        no_skill = len(y_true[y_true == 1]) / len(y_true)

        ax.plot([0, 1], [no_skill, no_skill], "k--", linewidth=4, label='random classifier')
        for model_id in model_ids:
            row = self.df[self.df['model_id'] == model_id].squeeze()
            if len(row['rr']) > 0:
                label = f"{model_id.split('_')[2].capitalize()} context (AUC = {row['auc_pr']:.2f})"
                ax.plot(row['rr'], row['pr'], linewidth=4, label=label)

        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("Precision", fontsize=16)
        ax.set_ylabel("Recall", fontsize=16)
        ax.set_title(f"Precision-Recall\n{model_ids[0].split('_')[0].upper()}", fontsize=18)
        ax.legend(loc="lower right")
        return ax

    def plot_confusion_matrix(self, model_ids: List[str], normalize=False, save_fig=False, path_save=None):
        model_ids = [model_ids] if isinstance(model_ids, str) else model_ids

        for model_id in model_ids:
            fig, ax = plt.subplots(figsize=(4, 4))

            model_data = self.df[self.df['model_id'] == model_id].squeeze()
            if model_data.empty:
                raise ValueError(f"No data found for model_id: {model_id}")

            cm = confusion_matrix(
                model_data['y_true'],
                model_data['y_pred']
            )

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title(f"{' '.join(model_id.split('_')[0:2])} {model_id.split('_')[2].capitalize()} Context", fontsize=16)
            plt.tight_layout()
            if save_fig:
                plt.savefig(join(path_save, f"{'_'.join(model_id.split('_')[0:3])}_confusion_matrix.pdf"), format='pdf', dpi=300)
            plt.show()
            plt.close(fig)

    def plot_metric_history(self, model_ids: List[str], metric: str, ax=None):
        model_ids = [model_ids] if isinstance(model_ids, str) else model_ids

        if not ax:
            fig, ax = plt.subplots(figsize=(10, 6))

        bar_containers = []

        for model_id in model_ids:
            filtered_df = self.df[self.df['model_id'] == model_id].squeeze()
            bars = ax.bar(
                f"{model_id.split('_')[2].capitalize()} context",
                filtered_df[metric],
                label=model_id
            )
            bar_containers.append(bars)

        for bars in bar_containers:
            ax.bar_label(bars, fmt='%.3f', padding=1)

        ax.set_title(f"{metric.upper()} Metric\n{' '.join(model_ids[0].split('_')[0:2])}")
        ax.set_ylabel(f"{metric.upper()} [%]")
        plt.xticks(size=8)
        return ax