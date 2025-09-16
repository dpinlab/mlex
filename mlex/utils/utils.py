import os
import numpy as np


def ensure_directory_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_first_middle_last_sequence_len(data_list):
    if len(data_list) < 3:
        return data_list
    middle_index = (len(data_list) - 1) // 2
    return [data_list[0], data_list[middle_index], data_list[-1]]


def make_json_serializable(params):
    import torch
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, torch.device):
            return str(v)
        elif isinstance(v, dict):
            return {k: convert(val) for k, val in v.items()}
        elif isinstance(v, (list, tuple)):
            return [convert(i) for i in v]
        else:
            return v
    return {k: convert(v) for k, v in params.items()}