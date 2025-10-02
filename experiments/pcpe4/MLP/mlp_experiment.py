import sys
from os.path import join, abspath
sys.path.append(abspath(join(__file__ , "..", "..", "..", "..")))
import torch
import numpy as np
import pandas as pd
from mlex import DataReader, MLP, F1MaxThresholdStrategy, StandardEvaluator
from sklearn.model_selection import GridSearchCV
from functools import partial
import pandas as pd
import itertools


path_train = r'/data/pcpe/pcpe_04.csv'
path_test = r'/data/pcpe/pcpe_03.csv'
target_column = 'I-d'
filter_data = {'NATUREZA_LANCAMENTO': 'C'}
threshold_strategy = 'f1max'
threshold_selection = F1MaxThresholdStrategy()

reader_train = DataReader(path_train, target_columns=[target_column], filter_dict=filter_data)
X_train = reader_train.fit_transform(X=None)
y_train = reader_train.get_target()

reader_test = DataReader(path_test, target_columns=[target_column], filter_dict=filter_data)
X_test = reader_test.fit_transform(X=None)
y_test = reader_test.get_target()

categories = [pd.unique(X_train[col]) for col in ['TIPO', 'CNAB', 'NATUREZA_SALDO']]


hidden_layer_sizes = [(10,), (20,), (10,10)]
learning_rate_init = [1e-2, 1e-3]
alpha = [0.0001, 0.001]
max_iter = [200]

combinations = list(itertools.product(hidden_layer_sizes, learning_rate_init, alpha, max_iter))

results_list = []

evaluator = StandardEvaluator(f"MLP_pipeline", threshold_selection)

for hl, lr, a, mi in combinations:
    print(f"Treinando MLP: hidden_layers={hl}, lr={lr}, alpha={a}, max_iter={mi}")
    mlp = MLP(target_column='I-d', categories=categories,
              hidden_layer_sizes=hl, learning_rate_init=lr,
              alpha=a, max_iter=mi, verbose=True)
    
    mlp.fit(X_train, y_train.values.flatten())
    
    y_scores = mlp.predict_proba(X_test)

    evaluator.evaluate(y_test.values.flatten(), [], y_scores)
    print(evaluator.summary())

    metrics = evaluator.results['metrics']
    results_list.append({
        'Hidden Layers': hl,
        'Learning Rate': lr,
        'Alpha': a,
        'Max Iter': mi,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        'AUC-PR': metrics['auc_pr'],
        'AUC-ROC': metrics['auc_roc']
    })
    



results = pd.DataFrame(results_list)

# # Função para gerar tabela LaTeX
# def generate_latex_table(df, caption="Resultados GridSearchCV", label="tab:grid_results"):
#     latex = df.to_latex(index=False, float_format="%.4f", caption=caption, label=label)
#     return latex

# latex_table = generate_latex_table(results)
# print(latex_table)

df = results.copy()

# Converter Learning Rate e Alpha para notação científica (10^x)
df['Learning Rate'] = df['Learning Rate'].apply(lambda x: f"$10^{{{int(np.log10(x))}}}$")
df['Alpha'] = df['Alpha'].apply(lambda x: f"$10^{{{int(np.log10(x))}}}$")

# Destacar melhor valor de cada métrica
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-PR', 'AUC-ROC']
for metric in metrics:
    max_val = df[metric].max()
    df[metric] = df[metric].apply(lambda x: f"\\textbf{{{x:.4f}}}" if x == max_val else f"{x:.4f}")

# Remover repetição de valores (apenas para visual)
for col in ['Hidden Layers', 'Learning Rate', 'Alpha', 'Max Iter']:
    prev = None
    for i in range(len(df)):
        if df[col].iloc[i] == prev:
            df[col].iloc[i] = ""
        else:
            prev = df[col].iloc[i]

# Gerar LaTeX centralizado
def generate_latex_table(df, caption="Resultados GridSearchCV", label="tab:grid_results"):
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format='c'*len(df.columns),
        caption=caption,
        label=label
    )
    #latex = latex.replace("\\begin{table}", "\\begin{table}[h!]\n\\centering")
    latex = "\\begin{table}[h!]\n\\centering\n\\small\n" + f"\\caption{{{caption}}}\n\\label{{{label}}}\n" + latex[6:] + "\\end{table}"

    return latex

latex_table = generate_latex_table(df)
print(latex_table)