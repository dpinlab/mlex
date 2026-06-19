"""

`HybridSVM`: pipeline RNN + LSTM + GRU → SVM.

Segue exatamente os mesmos padrões do repositório:
  - __init__ aceita kwargs flat (mesclando todos os schemas de parâmetros)

"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

from mlex.utils.context_aware import ContextAware
from mlex.utils.split import FeatureStratifiedSplit, PastFutureSplit
from mlex.models.base_components.recurrent.params import (
    RecurrentModelParams,
    PreprocessorParams,
    WrapperConfigParams,
)

from mlex.models.base_components.hybrid.hybrid_rnn_svm_params import HybridSVMParams
from mlex.models.base_components.hybrid.hybrid_rnn_svm_base import HybridRecurrentSVMBase


_PARAM_SCHEMAS = (WrapperConfigParams, RecurrentModelParams, PreprocessorParams, HybridSVMParams)


class HybridSVM(nn.Module, BaseEstimator, ClassifierMixin):
    """
    Modelo híbrido: três redes recorrentes (RNN, LSTM, GRU) cujos sinais
    são concatenados e enviados a um SVM, que aprende os pesos ótimos para
    a decisão final.

    """

    BASE_MODEL_CLS: type = None
    NAME: str = None

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------

    def __init__(self, **kwargs):
        super().__init__()
        (
            self.config,
            self.model_params,
            self.preprocessor_params,
            self.hybrid_params,
        ) = self._split_kwargs(kwargs)

        self.train_data = None
        self.fitted_ = False
        self._hybrid_base: HybridRecurrentSVMBase | None = None
        self.threshold: float = 0.5
        self.last_fit_time: float = 0.0
        self.categories: list | None = None


    @property
    def name(self): return self.NAME

    @property
    def target_column(self): return self.config.target_column

    @property
    def timestamp_column(self): return self.config.timestamp_column

    @property
    def context_column(self): return self.config.context_column

    @property
    def val_split(self): return self.config.val_split

    @property
    def split_stratify_column(self): return self.config.split_stratify_column

    @property
    def filter_dict(self): return self.config.filter_dict

    @property
    def sort_columns(self): return self.config.sort_columns

   

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "HybridSVM":
        """
        Orquestra toda a divisão dos dados e o treinamento do pipeline.

        Parameters
        ----------
        X : DataFrame com as features brutas (sem pré-processamento).
        y : Series com os rótulos binários.
        """
        self._update_params(kwargs)

        # 1. Filtro opcional
        if self.filter_dict is not None:
            X, y = self._apply_filter(X, y)

        # 2. Ordenação por contexto/tempo
        X, y = self._apply_context_aware_transform(X, y)

        # 3. Categorias para o pré-processador
        self._set_categories(X)

        # 4. Divisão em Treino-Redes / Treino-B-SVM
        X_rnn, y_rnn, X_svm_train, y_svm_train = self._split_rnn_svm(X, y)

        # 5. Dentro de X_rnn: Treino-A / Validação
        X_rnn_train, y_rnn_train, X_rnn_val, y_rnn_val = self._split_rnn_val(X_rnn, y_rnn)

        # 6. Instanciar e treinar o núcleo híbrido
        self._hybrid_base = HybridRecurrentSVMBase(
            model_params=self.model_params,
            preprocessor_params=self.preprocessor_params,
            hybrid_params=self.hybrid_params,
            target_column=self.target_column,
            timestamp_column=self.timestamp_column,
            context_column=self.context_column,
            categories=self.categories,
        )

        start = time.perf_counter()
        self._hybrid_base.fit(
            X_rnn_train, y_rnn_train,
            X_rnn_val,   y_rnn_val,
            X_svm_train, y_svm_train,
        )
        self.last_fit_time = time.perf_counter() - start

        self.threshold = self._hybrid_base.threshold
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------
    # predict_proba / predict
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Devolve a probabilidade da classe positiva para cada linha de X.

        Linhas que não compõem nenhuma janela completa (devido ao seq_length)
        recebem score 0.0.
        """
        self._validate_fitted()

        # Filtro e ordenação (mesma lógica do wrapper original)
        X_proc = self._apply_filter(X) if self.filter_dict is not None else X.copy()
        X_sorted, _ = self._apply_context_aware_transform(X_proc.copy(), None)

        # Pré-processamento
        preprocessor = self._hybrid_base.preprocessor_
        X_transformed = preprocessor.transform(X_sorted)

        # Sinais das redes → SVM → probabilidades (alinhadas às janelas)
        scores_windowed = self._hybrid_base.predict_proba_from_transformed(X_transformed)

        if len(scores_windowed) == 0:
            return np.zeros(len(X))

        # Recupera as posições das janelas (de qualquer rede, pois são iguais
        # após a interseção calculada em _extract_svm_features)
        first_model = next(iter(self._hybrid_base.recurrent_models_.values()))
        end_positions = np.array(first_model.predict_end_indices, dtype=int)

        # As posições da interseção são as mesmas usadas internamente
        common_positions = end_positions[:len(scores_windowed)]

        # Monta vetor alinhado ao X_sorted
        y_sorted = np.zeros(len(X_sorted), dtype=float)
        y_sorted[common_positions] = scores_windowed

        # Realinha para a ordem original de X
        return self._map_to_original_order(X, X_sorted, y_sorted)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._validate_fitted()
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ------------------------------------------------------------------
    # get_params / set_params (compatibilidade sklearn)
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        return {
            **self.config.model_dump(),
            **self.model_params.model_dump(),
            **self.preprocessor_params.model_dump(),
            **self.hybrid_params.model_dump(),
            'threshold': self.threshold,
        }

    def set_params(self, **parameters) -> "HybridSVM":
        if 'threshold' in parameters:
            self.threshold = parameters.pop('threshold')
        self._update_params(parameters)
        return self

    # ------------------------------------------------------------------
    # Utilitários internos (mesma lógica do _wrapper.py original)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_kwargs(kwargs: dict) -> tuple:
        """Distribui os kwargs entre os quatro schemas de parâmetros."""
        buckets = []
        leftover = dict(kwargs)
        for schema in _PARAM_SCHEMAS:
            schema_keys = schema.model_fields.keys()
            bucket = {k: leftover.pop(k) for k in list(leftover) if k in schema_keys}
            buckets.append(schema(**bucket))
        if leftover:
            raise TypeError(
                f"Parâmetros desconhecidos: {sorted(leftover)}. "
                f"Chaves válidas estão declaradas em "
                f"{', '.join(s.__name__ for s in _PARAM_SCHEMAS)}."
            )
        return tuple(buckets)

    def _update_params(self, updates: dict):
        for schema, attr in (
            (WrapperConfigParams,  'config'),
            (RecurrentModelParams, 'model_params'),
            (PreprocessorParams,   'preprocessor_params'),
            (HybridSVMParams,      'hybrid_params'),
        ):
            bucket = {k: v for k, v in updates.items() if k in schema.model_fields}
            if bucket:
                setattr(self, attr, getattr(self, attr).model_copy(update=bucket))

    def _apply_context_aware_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
    ) -> tuple:
        if self.timestamp_column is not None:
            sorter = ContextAware(
                target_column=self.target_column,
                timestamp_column=self.timestamp_column,
                context_column=self.context_column,
                sort_columns=self.sort_columns,
            )
            X, y = sorter.transform(X, y)
        return X, y

    def _apply_filter(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ):
        if self.filter_dict is None:
            return (X, y) if y is not None else X

        mask = pd.Series(True, index=X.index)
        for col, val in self.filter_dict.items():
            if col in X.columns:
                if isinstance(val, list):
                    mask &= X[col].isin(val)
                else:
                    mask &= (X[col] == val)

        X_filtered = X[mask].copy()
        if y is not None:
            return X_filtered, y[mask].copy()
        return X_filtered

    def _set_categories(self, X: pd.DataFrame):
        cat_feats = self.preprocessor_params.categorical_features
        if self.categories is None and cat_feats is not None:
            self.categories = [X[col].unique() for col in cat_feats]

    def _split_rnn_svm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Divide X/y em (Treino-Redes) e (Treino-B SVM) usando
        rnn_train_ratio (ex.: 0.7 / 0.3).
        """
        rnn_ratio = self.hybrid_params.rnn_train_ratio
        svm_ratio = 1.0 - rnn_ratio

        if self.hybrid_params.svm_split_stratify_column is not None:
            splitter = FeatureStratifiedSplit(
                stratify_column=self.hybrid_params.svm_split_stratify_column,
                split_proportion=svm_ratio,
            )
        else:
            splitter = PastFutureSplit(
                proportion=svm_ratio,
                timestamp_column=self.timestamp_column,
            )

        splitter.fit(X, y)
        # PastFutureSplit/FeatureStratifiedSplit devolvem
        # (X_train, y_train, X_val, y_val) onde "val" é a fatia futura/menor
        X_rnn, y_rnn, X_svm, y_svm = splitter.transform(X, y)
        return X_rnn, y_rnn, X_svm, y_svm

    def _split_rnn_val(
        self,
        X_rnn: pd.DataFrame,
        y_rnn: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Divide a fatia de Treino-Redes em Treino-A e Validação.

        rnn_val_ratio é relativo ao total original:
            val_proportion_within_rnn = rnn_val_ratio / rnn_train_ratio
        Exemplo: rnn_val_ratio=0.1, rnn_train_ratio=0.7 →
            dentro dos 70 %, separa 10/70 ≈ 14.3 % para validação.
        """
        val_proportion = (
            self.hybrid_params.rnn_val_ratio / self.hybrid_params.rnn_train_ratio
        )
        val_proportion = min(val_proportion, 0.9)   # salvaguarda

        if self.split_stratify_column is not None:
            splitter = FeatureStratifiedSplit(
                stratify_column=self.split_stratify_column,
                split_proportion=val_proportion,
            )
        else:
            splitter = PastFutureSplit(
                proportion=val_proportion,
                timestamp_column=self.timestamp_column,
            )

        splitter.fit(X_rnn, y_rnn)
        X_train, y_train, X_val, y_val = splitter.transform(X_rnn, y_rnn)
        return X_train, y_train, X_val, y_val

    def _map_to_original_order(
        self,
        X_original: pd.DataFrame,
        X_sorted: pd.DataFrame,
        y_pred_sorted: np.ndarray,
    ) -> np.ndarray:
        """Realinha predições (calculadas em X_sorted) para a ordem de X_original."""
        orig_indices = X_sorted['__orig_index'].values
        index_to_position = {idx: pos for pos, idx in enumerate(X_original.index)}

        y_pred = np.zeros(len(X_original), dtype=float)
        for sorted_pos, orig_idx in enumerate(orig_indices):
            if orig_idx in index_to_position:
                y_pred[index_to_position[orig_idx]] = y_pred_sorted[sorted_pos]
        return y_pred

    def _validate_fitted(self):
        if not self.fitted_:
            raise ValueError("HybridSVM: modelo não treinado. Chame fit() antes de predict.")