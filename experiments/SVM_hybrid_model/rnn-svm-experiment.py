"""
Pipeline para o modelo híbrido RNN + LSTM + GRU → SVM,

Divisão dos dados
-----------------
  path_train (pcpe_03.csv)
      ├── 60 % → Treino-A   : ajuste dos pesos das redes recorrentes
      ├── 10 % → Validação  : Early Stopping das redes
      └── 30 % → Treino-B  : treinamento do SVM

  path_test (pcpe_04.csv)
      └── 100 % → Avaliação final do pipeline completo
"""

import sys
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..", "..")))

import numpy as np
from mlex import hybrid_rnn_svm,DataReader, F1MaxThresholdStrategy, StandardEvaluator
from pcpe_utils import get_pcpe_dtype_dict, pcpe_preprocessing_read_func
import torch

path_train = r'/data/pcpe/pcpe_03.csv'
path_test  = r'/data/pcpe/pcpe_04.csv'

target_column      = 'I-d'
filter_data        = {'NATUREZA_LANCAMENTO': 'C'}
# sequence_column    = 'CONTA_TITULAR'          
# column_to_stratify = 'CPF_CNPJ_TITULAR'
num_layers=1
hidden_size=10
iterations=10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo selecionado para o treino das redes: {device}")

reader_train = DataReader(
    path_train,
    target_columns=[target_column],
    dtype_dict=get_pcpe_dtype_dict(),
    preprocessing_func=pcpe_preprocessing_read_func,
)
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(
    path_test,
    target_columns=[target_column],
    dtype_dict=get_pcpe_dtype_dict(),
    preprocessing_func=pcpe_preprocessing_read_func,
)
X_test, y_test = reader_test.get_X_y()


# rnn_train_ratio=0.7 → 70 % do treino vai para as redes recorrentes
# rnn_val_ratio=0.1   → desses 70 %, 10% são separados para validação
# Resultado: Treino-A=60 %, Validação=10 %, Treino-B(SVM)=30 %

sequence_lengths = [10, 20, 30, 40, 50]
for seq_len in sequence_lengths:
    print(f"Sequence Length: {seq_len}")
    for i in range(iterations):
        
        run_id = f"hybridrnnsvm_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{seq_len}_Iteration-{i+1}"

        model_hybrid = hybrid_rnn_svm(
            target_column=target_column,
            timestamp_column='DATA_LANCAMENTO',

            numeric_features=['DIA_LANCAMENTO', 'MES_LANCAMENTO', 'VALOR_TRANSACAO', 'VALOR_SALDO'],
            categorical_features=['TIPO', 'NATUREZA_SALDO'],
            # split_stratify_column=column_to_stratify,
        
            rnn_train_ratio=0.7,    # 70 % do treino para as redes
            rnn_val_ratio=0.1,      # 10 %(dos 70 %) para Early Stopping
            svm_kernel='rbf',
            svm_C=1.0,
            svm_gamma='scale',
            # context_column=sequence_column,
            filter_dict=filter_data,
            device=device,

            # --- Parâmetros das redes recorrentes (RecurrentModelParams) ---
            seq_length=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_size=32,  
            epochs=30,
            patience=5,

        
        
        )

        model_hybrid.fit(X_train, y_train)

        y_pred_score = model_hybrid.predict_proba(X_test)

        evaluator = StandardEvaluator(
            run_id,
            threshold=model_hybrid.threshold,
        )
        evaluator.evaluate(
            np.array(y_test.values.flatten()),
            [],
            y_pred_score,
        )

        print(evaluator.summary())
        print('\n')

        evaluator.save('evaluation_hybrid.parquet')
        evaluator.save('evaluation_hybrid.json')