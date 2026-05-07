from typing import Any, Optional
import torch
from pydantic import BaseModel, ConfigDict, Field


class RecurrentModelParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    input_size: Optional[int] = None
    hidden_size: int = Field(default=10, gt=0)
    num_layers: int = Field(default=1, gt=0)
    output_size: int = Field(default=1, gt=0)

    seq_length: int = Field(default=30, gt=0)
    batch_size: int = Field(default=32, gt=0)
    shuffle_dataloader: bool = True
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    persistent_workers: bool = False

    learning_rate: float = Field(default=1e-3, gt=0)
    alpha: float = 0.9
    eps: float = 1e-7
    weight_decay: float = 0.0
    epochs: int = Field(default=30, gt=0)
    patience: int = Field(default=5, ge=0)
    group_index: Optional[int] = -1
    random_seed: Optional[int] = 42

    device: Optional[torch.device] = None
    validation_data: Optional[tuple] = None
    collect_activations: bool = False
    dynamic_length_strategy: Optional[Any] = None
    dynamic_drop_last: bool = True
    epoch_observers: Optional[list] = None
    feature_names: Optional[Any] = None


class PreprocessorParams(BaseModel):
    model_config = ConfigDict(extra='forbid')

    numeric_features: Optional[list[str]] = None
    categorical_features: Optional[list[str]] = None
    passthrough_features: Optional[list[str]] = None
    context_feature: list[str] = Field(default_factory=lambda: ['CONTEXT'])


class WrapperConfigParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    target_column: str
    timestamp_column: str
    val_data: Optional[tuple] = None
    categories: Optional[list] = None
    context_column: Optional[str] = None
    val_split: float = Field(default=0.3, gt=0, lt=1)
    split_stratify_column: Optional[str] = None
    sort_columns: Optional[list[str]] = None
    filter_dict: Optional[dict] = None
