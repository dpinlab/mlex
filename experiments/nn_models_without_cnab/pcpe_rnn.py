import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import mlflow, mlflow.sklearn
import torch
import pandas as pd
import numpy as np
from mlex import RNN, DataReader, FeatureStratifiedSplit, F1MaxThresholdStrategy, StandardEvaluator, make_json_serializable, get_pcpe_dtype_dict, pcpe_preprocessing_read_func

EXPERIMENT = "fraud-detection_nn"
MODEL_NAME = "rnn_baseline_Id_without_cnab"
path_train = r'/data/pcpe/pcpe_03.csv'
path_test = r'/data/pcpe/pcpe_04.csv'

target_column = 'I-d'
filter_data = {'NATUREZA_LANCAMENTO': 'C'}
column_to_stratify = 'CPF_CNPJ_TITULAR'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT)


with mlflow.start_run(run_name="RNN_without_CNAB_v1"):
    reader_train = DataReader(path_train, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
    X, y = reader_train.get_X_y()

    reader_test = DataReader(path_test, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
    X_test, y_test = reader_test.get_X_y()

    X['GROUP'] = 'Unknown'
    X_test['GROUP'] = 'Unknown'

    splitter_tt = FeatureStratifiedSplit(column_to_stratify=column_to_stratify, test_proportion=0.3)
    splitter_tt.fit(X, y)
    X_train, y_train, X_val, y_val = splitter_tt.transform(X, y)

    categories = [pd.unique(X_train[col]) for col in ['TIPO', 'NATUREZA_SALDO']]

    validation_data = (X_val, y_val)
    model_RNN = RNN(validation_data=validation_data, target_column='I-d', categories=categories, device=device, categorical_features=['TIPO', 'NATUREZA_SALDO'], numeric_features=['DIA_LANCAMENTO','MES_LANCAMENTO','VALOR_TRANSACAO','VALOR_SALDO'], context_feature=['GROUP'])

    model_RNN.fit(X_train, y_train)

    y_pred_score = model_RNN.score_samples(X_test)

    y_true = model_RNN.get_y_true_sequences(X_test, y_test)

    evaluator = StandardEvaluator(f"RNN_pipeline", threshold_selection)
    evaluator.evaluate(np.array(y_true), [], y_pred_score)
    print(evaluator.summary())
    print('\n')

    print("\nLogging metrics and model to MLflow...")
    mlflow.log_metric("auc_roc", evaluator.results['metrics']['auc_roc'])
    mlflow.log_metric("f1", evaluator.results['metrics']['f1'])
    mlflow.log_metric("accuracy", evaluator.results['metrics']['accuracy'])
    mlflow.log_metric("precision", evaluator.results['metrics']['precision'])
    mlflow.log_metric("recall", evaluator.results['metrics']['recall'])
    mlflow.log_dict(make_json_serializable({k: v for k, v in model_RNN.get_params().items() if k != "validation_data"}), "model_params.json")
    mlflow.log_dict({'train_data': 'pcpe_03.csv', 'test_data': 'pcpe_04.csv', 'context':'baseline'}, "experimental_setup.json")
    mlflow.sklearn.log_model(model_RNN, name="model", registered_model_name=MODEL_NAME)
    print("--- Script Finished Successfully ---")