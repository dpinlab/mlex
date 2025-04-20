import matplotlib.pyplot as plt
import seaborn as sns
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
        filtered_df = self.df if model_ids is None else \
            self.df[self.df['model_id'].isin(model_ids)]

        ax.plot([0,1], [0,1], "k--",linewidth=4, label='random classifier')
        for _, row in filtered_df.iterrows():
            if len(row['metrics']['fpr']) > 0:
                label = f"{row['model_id']} (AUC = {row['metrics']['auc_roc']:.2f})"
                ax.plot(row['metrics']['fpr'], row['metrics']['tpr'],  linewidth=4, label=label)

        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=16)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=16)
        ax.set_title(f"Receiver Operating Characteristic \n ROC", fontsize=18)
        ax.legend(loc="lower right")
        return ax

    def plot_confusion_matrix(self, model_id: str, ax=None, normalize=False):
        if not ax:
            fig, ax = plt.subplots(figsize=(4, 4))

        model_data = self.df[self.df['model_id'] == model_id]
        if model_data.empty:
            raise ValueError(f"No data found for model_id: {model_id}")

        latest = model_data.sort_values('timestamp').iloc[-1]
        scalar_metrics = {
            k: v for k, v in latest['metrics'].items()
        }

        cm = confusion_matrix(
            scalar_metrics['y_true'],
            scalar_metrics['y_pred']
        )

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title(f"{model_id}", fontsize=18)
        plt.tight_layout()
        return ax

    def plot_metric_history(self, model_ids: List[str], metric: str, ax=None):
        if not ax:
            fig, ax = plt.subplots(figsize=(10, 6))
     
        filtered_df = self.df[self.df['model_id'].isin(model_ids)]

        for model_id, group in filtered_df.groupby('model_id'):
            ax.plot(
                group['timestamp'], 
                group['metrics'].apply(lambda x: x[metric]),
                'o-', 
                label=model_id
            )

        ax.set_title(f"{metric.upper()} History")
        ax.set_ylabel(metric.upper())
        ax.legend()
        plt.xticks(rotation=45)
        return ax