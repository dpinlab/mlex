import json
import os
from datetime import datetime

import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve

from .base import BaseEvaluator
from .utils import CustomEncoder


class StandardEvaluator(BaseEvaluator):
    def __init__(self, model_id, threshold_strategy=None):
        self.model_id = model_id
        self.threshold_strategy = threshold_strategy
        self.results = None
        self._schema = pa.schema([
            pa.field('timestamp', pa.timestamp('ns')),
            pa.field('model_id', pa.string()),
            pa.field('threshold', pa.float64()),
            pa.field('accuracy', pa.float64()),
            pa.field('precision', pa.float64()),
            pa.field('recall', pa.float64()),
            pa.field('f1', pa.float64()),
            pa.field('auc_pr', pa.float64()),
            pa.field('rr', pa.list_(pa.float64())),
            pa.field('pr', pa.list_(pa.float64())),
            pa.field('thresholds_pr', pa.list_(pa.float64())),
            pa.field('auc_roc', pa.float64()),
            pa.field('fpr', pa.list_(pa.float64())),
            pa.field('tpr', pa.list_(pa.float64())),
            pa.field('thresholds', pa.list_(pa.float64())),
            pa.field('y_true', pa.list_(pa.float64())),
            pa.field('y_pred', pa.list_(pa.float64())),
        ])

    def evaluate(self, y_true, y_pred, scores):
        binary = self._is_binary(y_true)
        threshold = None

        if binary and self.threshold_strategy:
            threshold = self.threshold_strategy.compute_threshold(y_true, scores)
            # if scores.ndim == 2 and scores.shape[1] == 2:
            #     scores = scores[:, 1]
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

        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        metrics.update({
            'auc_pr': auc(recall, precision),
            'rr': recall.tolist(),
            'pr': precision.tolist(),
            'thresholds_pr': thresholds.tolist()
        })

        if binary:
            # roc_scores = scores[:, 1] if scores.ndim == 2 else scores
            roc_scores = scores
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
        """Save results to file (Parquet or JSON) with append support"""
        if not self.results:
            raise ValueError("No results to save")

        # Prepare common data
        record = {
            'timestamp': self.results['timestamp'],
            'model_id': self.results['model_id'],
            'threshold': self.results['threshold'],
            'metrics': self.results['metrics']
        }

        if path.endswith('.parquet'):
            self._save_parquet(path, record)
        elif path.endswith('.json'):
            self._save_json(path, record)
        else:
            raise ValueError("Unsupported format. Use 'parquet' or 'json'")

    def _save_parquet(self, path, record):
        """Save/append to Parquet file"""
        # Convert to pyarrow types
        record['timestamp'] = pd.to_datetime(record['timestamp'])

        del record['metrics']
        record.update(self.results['metrics'])

        table = pa.Table.from_pylist([record], schema=self._schema)

        if os.path.exists(path):
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path)

    def _save_json(self, path, record):
        """Append `record` (a dict) into a JSON array on disk at `path`."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Invalid file path provided")

        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Load existing data if file exists, else start with empty list
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        # If it was a single dict, wrap it
                        data = [data]
                except json.JSONDecodeError:
                    # If file is empty or invalid, start fresh
                    data = []
        else:
            data = []

        # Append the new record
        data.append(record)

        # Write back the full list
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=CustomEncoder, indent=2)
            f.write('\n')

    def load(self, path):
        """Load from either Parquet or JSON file"""
        if path.endswith('.parquet'):
            self._load_parquet(path)
        elif path.endswith('.json'):
            self._load_json(path)
        else:
            raise ValueError("Unsupported file format")

    def _load_parquet(self, path):
        """Load from Parquet file"""
        table = pq.read_table(path)
        df = table.to_pandas()
        self.results = df.iloc[-1].to_dict()  # Get latest record
        self._convert_timestamp()

    def _load_json(self, path):
        """Load from JSON file (gets latest record)"""
        with open(path, 'r') as f:
            lines = f.readlines()
            if not lines:
                raise ValueError("Empty JSON file")
            self.results = json.loads(lines[-1])
        self._convert_timestamp()

    def _convert_timestamp(self):
        """Ensure timestamp is in ISO format"""
        if 'timestamp' in self.results:
            self.results['timestamp'] = pd.to_datetime(
                self.results['timestamp']
            ).isoformat()

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


