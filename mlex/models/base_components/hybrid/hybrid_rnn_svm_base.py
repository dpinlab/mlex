"""
==============
Núcleo do pipeline híbrido Redes Recorrentes → SVM.

Fluxo de dados
--------------
                        ┌─────────┐
       Treino-A ────────► RNN fit │
                        │ LSTM fit│
                        │ GRU fit │
                        └────┬────┘
                             │  (Early Stopping via Validação)
                        ┌────▼─────────────────────────────────┐
       Treino-B ────────► RNN.predict │ LSTM.predict │ GRU.pred │
                        │        concatenar sinais             │
                        └────────────────┬─────────────────────┘
                                         │
                                    ┌────▼────┐
                                    │ SVM fit │
                                    └────┬────┘
                                         │
                             predict_proba (Teste)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from mlex.models.base_components.recurrent.concrete import (
    RNNBaseModel,
    LSTMBaseModel,
    GRUBaseModel,
)
from mlex.models.base_components.recurrent.params import (
    RecurrentModelParams,
    PreprocessorParams,
)
from mlex.utils.preprocessing import PreProcessingTransformer
from mlex.utils.split import FeatureStratifiedSplit, PastFutureSplit
from mlex.evaluation.threshold import F1MaxThresholdStrategy

from mlex.models.base_components.hybrid.hybrid_rnn_svm_params import HybridSVMParams


# ---------------------------------------------------------------------------
# Mapeamento nome → classe base de rede recorrente
# ---------------------------------------------------------------------------
_RECURRENT_REGISTRY: dict[str, type] = {
    'rnn':  RNNBaseModel,
    'lstm': LSTMBaseModel,
    'gru':  GRUBaseModel,
}


class HybridRecurrentSVMBase:
    """
    Classe base que orquestra:
      1. Pré-processamento compartilhado (PreProcessingTransformer).
      2. Treinamento independente de N redes recorrentes em Treino-A
         com Early Stopping em Validação.
      3. Extração dos sinais de saída de cada rede em Treino-B.
      4. Concatenação dos sinais → SVM treinado em Treino-B.
      5. Inferência final: sinais das redes concatenados → SVM.predict_proba.

    
    """

    # Subclasses podem sobrescrever para usar apenas algumas arquiteturas.
    ARCHITECTURES: tuple[str, ...] = ('rnn', 'lstm', 'gru')

    def __init__(
        self,
        model_params: RecurrentModelParams,
        preprocessor_params: PreprocessorParams,
        hybrid_params: HybridSVMParams,
        target_column: str,
        timestamp_column: str,
        context_column: str | None,
        categories: list | None,
    ):
        self.model_params = model_params
        self.preprocessor_params = preprocessor_params
        self.hybrid_params = hybrid_params
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.context_column = context_column
        self.categories = categories

        # Instâncias criadas durante fit()
        self.preprocessor_: PreProcessingTransformer | None = None
        self.recurrent_models_: dict[str, object] = {}
        self.svm_: SVC | None = None
        self.threshold: float = 0.5
        self.fitted_: bool = False

  

    def fit(
        self,
        X_rnn_train: pd.DataFrame,
        y_rnn_train: pd.Series,
        X_rnn_val: pd.DataFrame,
        y_rnn_val: pd.Series,
        X_svm_train: pd.DataFrame,
        y_svm_train: pd.Series,
    ) -> "HybridRecurrentSVMBase":
        """
        Recebe os três conjuntos já separados pelo wrapper e executa
        todo o pipeline de treinamento.

        Parameters
        ----------
        X_rnn_train, y_rnn_train : Treino-A  — ajuste dos pesos das redes
        X_rnn_val,   y_rnn_val   : Validação  — Early Stopping
        X_svm_train, y_svm_train : Treino-B  — treinamento do SVM
        """
        # 1. Ajuste do pré-processador em Treino-A
        self.preprocessor_ = self._build_preprocessor()
        self.preprocessor_.fit(X_rnn_train)
        feature_names = self.preprocessor_.get_feature_names_out()
        input_size = feature_names.shape[0] - 1   # exclui coluna-alvo

        # 2. Transformar os três conjuntos
        X_rnn_train_t, y_rnn_train_t = self.preprocessor_.transform(X_rnn_train, y_rnn_train)
        X_rnn_val_t,   y_rnn_val_t   = self.preprocessor_.transform(X_rnn_val,   y_rnn_val)
        X_svm_train_t, _             = self.preprocessor_.transform(X_svm_train, y_svm_train)

        # 3. Treinar cada rede recorrente em Treino-A (val → Early Stopping)
        for arch_name in self.ARCHITECTURES:
            model = self._build_recurrent(arch_name, input_size, X_rnn_val_t, y_rnn_val_t)
            model.fit(X_rnn_train_t, y_rnn_train_t)
            self.recurrent_models_[arch_name] = model

        # 4. Extrair sinais em Treino-B e treinar SVM
        svm_features, common_pos_train = self._extract_svm_features(X_svm_train_t)
        svm_labels = y_svm_train.values[common_pos_train].astype(int)

        self.svm_ = SVC(
            kernel=self.hybrid_params.svm_kernel,
            C=self.hybrid_params.svm_C,
            gamma=self.hybrid_params.svm_gamma,
            probability=self.hybrid_params.svm_probability,
        )
        self.svm_.fit(svm_features, svm_labels)

        # 5. Threshold via F1-max nos dados de validação das redes
        val_features, common_pos_val = self._extract_svm_features(X_rnn_val_t)
        val_proba    = self.svm_.predict_proba(val_features)[:, 1]
        val_labels   = y_rnn_val.values[common_pos_val].astype(int)

        # Alinha tamanhos (redes só produzem saída para janelas completas)
        self.threshold = F1MaxThresholdStrategy().compute_threshold(
            val_labels, val_proba
        )

        self.fitted_ = True
        return self

    def predict_proba_from_transformed(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Recebe dados já pré-processados e devolve as probabilidades do SVM.
        Chamado pelo wrapper após aplicar o preprocessor_ em X_test.
        """
        self._check_fitted()
        features, _ = self._extract_svm_features(X_transformed)
        if features.shape[0] == 0:
            return np.array([])
        return self.svm_.predict_proba(features)[:, 1]

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _build_preprocessor(self) -> PreProcessingTransformer:
        return PreProcessingTransformer(
            target_column=[self.target_column],
            **self.preprocessor_params.model_dump(),
            categories=self.categories,
            handle_unknown='ignore',
        )

    def _build_recurrent(
        self,
        arch_name: str,
        input_size: int,
        X_val_t,
        y_val_t,
    ):
        """Instancia e devolve um modelo recorrente ainda não treinado."""
        cls = _RECURRENT_REGISTRY[arch_name]
        params_dict = self.model_params.model_dump()
        params_dict.update({
            'input_size': input_size,
            'validation_data': (X_val_t, y_val_t),
        })
        return cls(**params_dict)

    def _extract_svm_features(self, X_transformed) -> np.ndarray:
        """
        Roda predict() em cada rede para X_transformed e concatena os sinais.

        Cada rede devolve uma lista de scores apenas para as posições
        correspondentes ao fim de cada janela (seq_length..N).
        O conjunto de posições é determinado pelo dataset interno de cada rede.
        Usamos a interseção das posições válidas entre todas as redes para
        garantir que o SVM sempre receba vetores alinhados.
        """
        signals: dict[str, np.ndarray] = {}
        end_positions: dict[str, np.ndarray] = {}

        for name, model in self.recurrent_models_.items():
            preds = np.array(model.predict(X_transformed)).flatten()
            pos   = np.array(model.predict_end_indices, dtype=int)
            signals[name]      = preds
            end_positions[name] = pos

        # Interseção das posições válidas
        common_positions = set(end_positions[self.ARCHITECTURES[0]])
        for arch in self.ARCHITECTURES[1:]:
            common_positions &= set(end_positions[arch])
        common_positions = np.array(sorted(common_positions), dtype=int)

        if len(common_positions) == 0:
            return np.empty((0, len(self.ARCHITECTURES))), np.array([], dtype=int)

        # Monta a matriz [n_samples, n_architectures]
        columns = []
        for name in self.ARCHITECTURES:
            pos   = end_positions[name]
            preds = signals[name]
            pos_to_score = dict(zip(pos, preds))
            col = np.array([pos_to_score[p] for p in common_positions])
            columns.append(col)

        # RETORNE TAMBÉM AS POSIÇÕES VÁLIDAS
        return np.column_stack(columns), common_positions # shape: (n_windows, n_archs)

    def _check_fitted(self):
        if not self.fitted_:
            raise ValueError("HybridRecurrentSVMBase: modelo não foi treinado. Chame fit() primeiro.")