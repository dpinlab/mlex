import sys
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..", "..", "..","..")))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from mlex import DataReader, RandomForest, F1MaxThresholdStrategy, StandardEvaluator, PastFutureSplit
import itertools
import pandas as pd

path = r'/data/eeg_eyestate/EEG_Eye_State.csv'

target_column = 'eyeDetection'
filter_data = {}
threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()

reader = DataReader(path, target_columns=[target_column] )
X_full,y_full = reader.get_X_y()


X_full = X_full.drop(columns=["Timestamp"])

y_full[target_column] = y_full[target_column].astype(int)


splitter_tt = PastFutureSplit(proportion=0.75)
splitter_tt.fit(X_full, y_full)
X_train_full, y_train_full, X_test, y_test = splitter_tt.transform(X_full, y_full)

# X_train_full['GROUP'] = 'Unknown'
# X_test['GROUP'] = 'Unknown'



y_train_full[target_column] = y_train_full[target_column].astype(int)
y_test[target_column] = y_test[target_column].astype(int)



n_estimators = [50, 100, 200]
max_depth = [None, 5, 10]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
combinations = list(itertools.product(n_estimators, max_depth, min_samples_split, min_samples_leaf))


results_list = []

evaluator = StandardEvaluator("RF_GridSearch", threshold_selection)

for n, depth, split, leaf in combinations:
    print(f"Treinando RandomForest: n_estimators={n}, max_depth={depth}, min_samples_split={split}, min_samples_leaf={leaf}")

    model_RF = RandomForest(target_column='I-d', categories=[],
        n_estimators=n,
        max_depth=depth,
        min_samples_split=split,
        min_samples_leaf=leaf,
        verbose=True,
        numeric_features= X_train_full.columns.to_list(), 
        categorical_features=[]  )

    model_RF.fit(X_train_full, y_train_full.values.flatten())

    y_pred_score = model_RF.predict_proba(X_test)


    evaluator.evaluate(y_test.values.flatten(), [], y_pred_score)
    print(evaluator.summary())
    print('\n')

    print('Feature importances:')
    print(model_RF.feature_importances())
    print('\n')

    print('Permutation importances:')
    print(model_RF.permutation_importances(X_test, y_test, n_repeats=10))
    print('\n')

    metrics = evaluator.results['metrics']
    results_list.append({
        'n_estimators': n,
        'max_depth': depth,
        'min_samples_split': split,
        'min_samples_leaf': leaf,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        'AUC-PR': metrics['auc_pr'],
        'AUC-ROC': metrics['auc_roc']
    })


results_df = pd.DataFrame(results_list)
df = results_df.copy()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-PR', 'AUC-ROC']
for metric in metrics:
    max_val = df[metric].max()
    df[metric] = df[metric].apply(lambda x: f"\\textbf{{{x:.4f}}}" if x == max_val else f"{x:.4f}")

# Remover repetição de valores (apenas para visual)
for col in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
    prev = None
    for i in range(len(df)):
        if df[col].iloc[i] == prev:
            df[col].iloc[i] = ""
        else:
            prev = df[col].iloc[i]

# Gerar LaTeX centralizado
def generate_latex_table(df, caption="Resultados GridSearch RandomForest", label="tab:rf_grid_results"):
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format='c'*len(df.columns),
        caption=caption,
        label=label
    )
    latex = "\\begin{table}[h!]\n\\centering\n\\small\n" + f"\\caption{{{caption}}}\n\\label{{{label}}}\n" + latex[6:] + "\\end{table}"
    return latex

latex_table = generate_latex_table(df)
print(latex_table)
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