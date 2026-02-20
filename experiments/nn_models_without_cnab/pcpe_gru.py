import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import mlflow
from mlflow import sklearn as mlflowsklearn
import numpy as np
from mlex import GRU, DataReader, F1MaxThresholdStrategy, StandardEvaluator, make_json_serializable
from pcpe_utils import get_pcpe_dtype_dict, pcpe_preprocessing_read_func

EXPERIMENT = "fraud-detection_nn"
MODEL_NAME = "gru_baseline_Id_without_cnab"
path_train = r'/data/pcpe/pcpe_03.csv'
path_test = r'/data/pcpe/pcpe_04.csv'

target_column = 'I-d'
filter_data = {'NATUREZA_LANCAMENTO': 'C'}
column_to_stratify = 'CPF_CNPJ_TITULAR'
threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="GRU_without_CNAB_v1"):
    reader_train = DataReader(path_train, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
    X, y = reader_train.get_X_y()

    reader_test = DataReader(path_test, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
    X_test, y_test = reader_test.get_X_y()

    model_GRU = GRU(
        target_column=target_column,
        categorical_features=['TIPO', 'NATUREZA_SALDO'],
        numeric_features=['DIA_LANCAMENTO','MES_LANCAMENTO','VALOR_TRANSACAO','VALOR_SALDO'],
        split_stratify_column=column_to_stratify,
        val_split=0.3,
        context_column=None,
        timestamp_column='DATA_LANCAMENTO'
    )

    model_GRU.fit(X, y)

    y_pred_score = model_GRU.predict_proba(X_test)

    model_GRU.set_params(threshold = threshold_selection.compute_threshold(y_test, y_pred_score))

    evaluator = StandardEvaluator(f"GRU_pipeline", threshold=model_GRU.threshold)
    evaluator.evaluate(np.array(y_test.values.flatten()), [], y_pred_score)
    print(evaluator.summary())
    print('\n')

    print("\nLogging metrics and model to MLflow...")
    mlflow.log_metric("auc_roc", evaluator.results['metrics']['auc_roc'])
    mlflow.log_metric("f1", evaluator.results['metrics']['f1'])
    mlflow.log_metric("accuracy", evaluator.results['metrics']['accuracy'])
    mlflow.log_metric("precision", evaluator.results['metrics']['precision'])
    mlflow.log_metric("recall", evaluator.results['metrics']['recall'])
    mlflow.log_dict(make_json_serializable({k: v for k, v in model_GRU.get_params().items()}), "model_params.json")
    mlflow.log_dict({'train_data': 'pcpe_03.csv', 'test_data': 'pcpe_04.csv', 'context':'baseline'}, "experimental_setup.json")
    mlflowsklearn.log_model(model_GRU, name="model", registered_model_name=MODEL_NAME)
    print("--- Script Finished Successfully ---")
