import sys
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..", "..")))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from mlex import DataReader, RandomForest, F1MaxThresholdStrategy, StandardEvaluator
from pcpe_utils import get_pcpe_dtype_dict, pcpe_preprocessing_read_func

path_train = r'/data/pcpe/pcpe_03.csv'
path_test = r'/data/pcpe/pcpe_04.csv'
target_column = 'I-d'
filter_data = {'NATUREZA_LANCAMENTO': 'C'}
threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()

reader_train = DataReader(path_train, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
X_train, y_train = reader_train.get_X_y()

reader_test = DataReader(path_test, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
X_test, y_test = reader_test.get_X_y()

categories = [pd.unique(X_train[col]) for col in ['TIPO', 'CNAB', 'NATUREZA_SALDO']]

model_RF = RandomForest(target_column='I-d', categories=categories, numeric_features=['DIA_LANCAMENTO','MES_LANCAMENTO','VALOR_TRANSACAO','VALOR_SALDO'], categorical_features=['TIPO', 'CNAB', 'NATUREZA_SALDO'])

model_RF.fit(X_train, y_train.values.flatten())

y_pred_score = model_RF.predict_proba(X_test)

print('\n')
evaluator = StandardEvaluator(f"RF_pipeline", threshold_selection)
evaluator.evaluate(y_test.values.flatten(), [], y_pred_score)
print(evaluator.summary())
print('\n')

print('Feature importances:')
print(model_RF.feature_importances())
print('\n')

print('Permutation importances:')
print(model_RF.permutation_importances(X_test, y_test, n_repeats=10))
print('\n')

# indicators, index_by_tree = model_RF.decision_path(X_test.iloc[[0]])
# indices = zip(index_by_tree, index_by_tree[1:])
# for tree_classifier, (begin, end) in zip(model_RF.model.named_steps['final_model'].estimators_, indices):
#     tree = tree_classifier.tree_
#     node_indices = indicators[0, begin:end].indices


# tree_to_visualize = model_RF.model.named_steps['final_model'].estimators_[0]
# plt.figure(figsize=(20, 10)) # Adjust figure size for better readability
# plot_tree(tree_to_visualize,
#                 feature_names=model_RF.get_feature_names(),
#                 class_names=["0", "1"],
#                 filled=True,
#                 rounded=True)
# plt.title("Decision Tree from Random Forest")
# plt.show()
# plt.savefig('random_forest_tree.pdf', format='pdf')
# print()


# evaluator.save('evaluation.parquet')
# evaluator.save('evaluation.json')
# model_RF.model