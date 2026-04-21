from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClassicalWrapperConfigParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    target_column: str
    categories: Optional[list] = None
    filter_dict: Optional[dict] = None


class ClassicalPreprocessorParams(BaseModel):
    model_config = ConfigDict(extra='forbid')

    numeric_features: Optional[list[str]] = None
    categorical_features: Optional[list[str]] = None
    passthrough_features: Optional[list[str]] = None
    context_feature: Optional[list[str]] = None


class MLPParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    hidden_layer_sizes: Any = (10,)
    activation: str = 'relu'
    solver: str = 'adam'
    batch_size: Any = 32
    shuffle: bool = True
    learning_rate: str = 'constant'
    learning_rate_init: float = Field(default=1e-3, gt=0)
    alpha: float = Field(default=1e-4, ge=0)
    epsilon: float = Field(default=1e-8, gt=0)
    max_iter: int = Field(default=100, gt=0)
    random_state: Optional[int] = None
    validation_fraction: float = Field(default=0.3, gt=0, lt=1)
    early_stopping: bool = True
    verbose: bool = True


class RandomForestParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    n_estimators: int = Field(default=100, gt=0)
    criterion: str = 'gini'
    max_depth: Optional[int] = None
    min_samples_split: Any = 2
    min_samples_leaf: Any = 1
    min_weight_fraction_leaf: float = Field(default=0.0, ge=0.0)
    max_features: Any = 'sqrt'
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = Field(default=0.0, ge=0.0)
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    verbose: Any = True
    warm_start: bool = False
    class_weight: Any = None
    ccp_alpha: float = Field(default=0.0, ge=0.0)
    max_samples: Any = None
    monotonic_cst: Any = None
