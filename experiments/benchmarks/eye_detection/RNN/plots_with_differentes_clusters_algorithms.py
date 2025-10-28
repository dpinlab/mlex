import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
from os.path import join
import matplotlib.pyplot as plt
from mlex.evaluation.plotter import EvaluationPlotter
from itertools import product


def ensure_directory_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


models    = ['RNN']
sequences = ['feature', 'temporal']
lengths   = ['10', '20', '30', '40', '50']
thresholds_list = ['f1max']
iterations = 10
iteration = 10
num_layers = 1
hidden_size = 10
clusters = ['kmeans', 'gmm', 'agglomerative']

save_path = join("results", f"{num_layers}-layer")
ensure_directory_exists(save_path)
plotter = EvaluationPlotter(f"results/evaluation.parquet")

# Generate the strings


curves = [
    {"sequence": "temporal", "cluster": "kmeans", "label": "KMeans - Temporal"},
    {"sequence": "feature",  "cluster": "kmeans", "label": "KMeans - Feature"},
    {"sequence": "temporal", "cluster": "gmm", "label": "GMM - Temporal"},
    {"sequence": "feature",  "cluster": "gmm", "label": "GMM - Feature"},
    {"sequence": "temporal", "cluster": "agglomerative", "label": "Agglomerative - Temporal"},
    {"sequence": "feature",  "cluster": "agglomerative", "label": "Agglomerative - Feature"},
]

curve_model_ids = []

for curve in curves:
    seq = curve["sequence"]
    cl = curve["cluster"]
    ids = []
    for model, length, threshold, i in product(models, lengths, thresholds_list, range(iterations)):
        model_id = f"{model}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{length}_{seq}_{threshold}_{cl}"
        ids.append(model_id)
    curve_model_ids.append(ids)

for length in lengths:
    string_plot = f"{models[0]}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{length}_clusters_combined"

    # Preparar os grupos filtrando pelos model_ids do length atual
    groups = []
    labels = []
    for curve, model_ids in zip(curves, curve_model_ids):
        filtered_ids = [mid for mid in model_ids if f"SequenceLength-{length}_" in mid]
        groups.append(filtered_ids)
        labels.append(curve["label"])

    # Plot ROC curve com todas as curvas numa única chamada
    fig, ax = plt.subplots()
    fixed_colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                '#d62728', '#9467bd', '#8c564b']

    labels = [c["label"] for c in curves]

    plotter.plot_roc_curve_with_ci(groups, ax=ax,colors=fixed_colors, labels = labels,shade=False)
    #ax.legend(labels, loc="lower right")
    ax.set_title(f"ROC curves with 95% CI {models[0]} - SequenceLength {length}")
    plt.savefig(join(save_path, f"{string_plot}_roc_curve_with_ci.pdf"), format="pdf", dpi=300)
    plt.show()
    plt.close(fig)

    # Plot histórico F1 com todas as curvas numa única chamada
    fig, ax = plt.subplots()
    plotter.plot_metric_history_with_ci(model_groups=groups, metric="f1", ax=ax)
    ax.legend(labels)
    ax.set_title(f"{models[0]}: F1 with 95% CI - SequenceLength {length}")
    plt.savefig(join(save_path, f"{string_plot}_f1_metric_history_with_ci.pdf"), format="pdf", dpi=300)
    plt.show()
    plt.close(fig)





