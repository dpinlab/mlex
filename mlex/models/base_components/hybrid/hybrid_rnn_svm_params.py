from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class HybridSVMParams(BaseModel):
    """
    Parâmetros exclusivos do classificador SVM e da divisão de dados usada
    no treinamento do pipeline híbrido (Redes Recorrentes + SVM).
    """
    model_config = ConfigDict(extra='forbid')

    # --- Divisão interna do treino ---
    # Proporção dos dados de treino destinada ao ajuste das redes recorrentes.
    # O complemento (1 - rnn_train_ratio) é reservado para o Treino-B do SVM.
    rnn_train_ratio: float = Field(default=0.7, gt=0, lt=1)

    # Dentro da fatia destinada às redes, qual proporção vira validação
    # Exemplo: rnn_train_ratio=0.7, rnn_val_ratio=0.1
    #   → Treino-RNN = 60 %, Validação-RNN = 10 %, Treino-SVM = 30 %
    rnn_val_ratio: float = Field(default=0.1, gt=0, lt=1)

    # --- Hiperparâmetros do SVM ---
    svm_kernel: str = Field(default='rbf')
    svm_C: float = Field(default=1.0, gt=0)
    svm_gamma: str = Field(default='scale')     
    svm_probability: bool = True                #  predict_proba

    # Coluna usada para estratificar a divisão Treino-RNN / Validação-RNN / Treino-SVM.
    # Se None, usa corte temporal simples (PastFutureSplit).
    svm_split_stratify_column: Optional[str] = None