import os
from datetime import datetime

import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

from .base import BaseEvaluator


class StandardEvaluator(BaseEvaluator):
    def __init__(self, model_id, threshold_strategy=None):
        self.model_id = model_id
        self.threshold_strategy = threshold_strategy
        self.results = None
        self._schema = pa.schema([
            pa.field('timestamp', pa.timestamp('ns')),
            pa.field('model_id', pa.string()),
            pa.field('threshold', pa.float64()),
            pa.field('metrics', pa.struct([
                ('accuracy', pa.float64()),
                ('precision', pa.float64()),
                ('recall', pa.float64()),
                ('f1', pa.float64()),
                ('auc_roc', pa.float64()),
                ('fpr', pa.list_(pa.float64())),
                ('tpr', pa.list_(pa.float64())),
                ('thresholds', pa.list_(pa.float64())),
                ('y_true', pa.list_(pa.float64())),
                ('y_pred', pa.list_(pa.float64()))
            ]))
        ])

    def evaluate(self, y_true, y_pred, scores):
        binary = self._is_binary(y_true)
        threshold = None

        if binary and self.threshold_strategy:
            threshold = self.threshold_strategy.compute_threshold(y_true, scores)
            if scores.ndim == 2 and scores.shape[1] == 2:
                scores = scores[:, 1]
            y_pred = (scores >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred,
                        average='binary' if binary else 'macro'),
            'recall': recall_score(y_true, y_pred,
                        average='binary' if binary else 'macro'),
            'f1': f1_score(y_true, y_pred,
                        average='binary' if binary else 'macro')
        }

        if binary:
            roc_scores = scores[:, 1] if scores.ndim == 2 else scores
            fpr, tpr, thresholds = roc_curve(y_true, roc_scores)
            metrics.update({
                'auc_roc': auc(fpr, tpr),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            })
        else:
            metrics['auc_roc'] = roc_auc_score(y_true, scores,
                                              multi_class='ovr',
                                              average='macro')
            metrics.update({
                'fpr': [],
                'tpr': [],
                'thresholds': []
            })

        metrics.update({
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist()
        })

        self.results = {
            'timestamp': datetime.now().isoformat(),
            'model_id': self.model_id,
            'threshold': float(threshold) if threshold is not None else None,
            'metrics': metrics
        }

    def save(self, path):
        if not self.results:
            raise ValueError("No results to save")

        new_data = {
            'timestamp': pd.to_datetime(self.results['timestamp']),
            'model_id': self.results['model_id'],
            'threshold': self.results['threshold'],
            'metrics': self.results['metrics']
        }
        new_df = pd.DataFrame([new_data])
        new_table = pa.Table.from_pandas(new_df, schema=self._schema)

        if os.path.exists(path):
            existing_table = pq.read_table(path)

            if existing_table.schema != self._schema:
                raise ValueError("Schema mismatch with existing Parquet file")

            combined_table = pa.concat_tables([existing_table, new_table])
            pq.write_table(combined_table, path)
        else:
            pq.write_table(new_table, path)

    def load(self, path):
        table = pq.read_table(path)
        df = table.to_pandas().sort_values('timestamp', ascending=False)
        if not df.empty:
            self.results = df.iloc[0].to_dict()
            self.results['timestamp'] = self.results['timestamp'].isoformat()

    def summary(self):
        if not self.results:
            return "No evaluation results available"

        lines = [
            f"Evaluation Summary - {self.results['model_id']}",
            f"Timestamp: {self.results['timestamp']}",
        ]

        if self.results['threshold'] is not None:
            lines.append(f"Optimal Threshold: {self.results['threshold']:.4f}")

        scalar_metrics = {k: v for k, v in self.results['metrics'].items()
                        if isinstance(v, (int, float))}
        metrics = "\n".join([f"{k}: {v:.4f}" for k, v in scalar_metrics.items()])
        lines.append(metrics)
        return "\n".join(lines)

    @staticmethod
    def parquet_summary(file_path, model_id=None):
        """Read and summarize Parquet evaluation records with optional model filtering"""
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()

            if model_id:
                df = df[df['model_id'] == model_id]

            if df.empty:
                return "No matching records found"

            df = df.sort_values('timestamp')
            summaries = []

            for _, row in df.iterrows():
                summary = [
                    f"Model ID: {row['model_id']}",
                    f"Timestamp: {row['timestamp']}",
                    f"Threshold: {row['threshold']:.4f}",
                    "Metrics:"
                ]

                scalar_metrics = {
                    k: v for k, v in row['metrics'].items()
                    if isinstance(v, (int, float))
                }
                metrics = "\n".join(
                    [f"  {k}: {v:.4f}" for k, v in scalar_metrics.items()]
                )

                if len(row['metrics']['fpr']) > 0:
                    metrics += "\n  ROC Curve Points: {}".format(
                        len(row['metrics']['fpr'])
                    )

                summaries.append("\n".join(summary + [metrics]))

            return "\n\n".join(summaries)

        except Exception as e:
            return f"Error reading Parquet file: {str(e)}"

    @staticmethod
    def get_roc_data(file_path, model_id=None):
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()

            if model_id:
                df = df[df['model_id'] == model_id]

            return {
                'fpr': df['metrics'].apply(lambda x: x['fpr']).tolist(),
                'tpr': df['metrics'].apply(lambda x: x['tpr']).tolist(),
                'thresholds': df['metrics'].apply(lambda x: x['thresholds']).tolist(),
                'auc': df['metrics'].apply(lambda x: x['auc_roc']).tolist()
            }
        except Exception as e:
            print(f"Error retrieving ROC data: {str(e)}")
            return None
