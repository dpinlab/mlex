import json
import numpy as np


def compare_evaluations(eval1, eval2):
    comparison = {}
    for metric in eval1['metrics']:
        if metric in eval2['metrics']:
            comparison[metric] = eval2['metrics'][metric] - eval1['metrics'][metric]
    return comparison


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)