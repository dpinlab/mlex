import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
from os.path import join
import matplotlib.pyplot as plt
from mlex.evaluation.plotter import EvaluationPlotter
from itertools import product


def ensure_directory_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


models    = ['RNN']
sequences = ['temporal', 'account', 'individual']
lengths   = ['10', '20', '30', '40', '50']
thresholds_list = ['f1max']
iteration = 10
num_layers = 1
hidden_size = 2

save_path = join("results", f"{num_layers}-layer")
ensure_directory_exists(save_path)
plotter = EvaluationPlotter(f"evaluation.parquet")

# Generate the strings
formatted_ids = [
    f"{model}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{length}_{sequence}_{threshold}_Iteration-{iteration}"
    for model, length, sequence, threshold in product(models, lengths, sequences, thresholds_list)
]

list_model_ids = [formatted_ids[i:i+len(sequences)] for i in range(0, len(formatted_ids), len(sequences))]

for model_ids in list_model_ids:
    # string_plot = f"{'_'.join(model_ids[0].split('_')[0:2])}_{num_layers}-layer"
    string_plot = model_ids[0][0].rsplit("_", 3)[0]
    fig, ax = plt.subplots()
    plotter.plot_roc_curve(model_ids, ax=ax)
    plt.savefig(join(save_path, f"{string_plot}_cbic2025-roc_curve.pdf"), format="pdf", dpi=300)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    plotter.plot_pr_curve(model_ids, ax=ax)
    plt.savefig(join(save_path, f"{string_plot}_pr_curve.pdf"), format="pdf", dpi=300)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    plotter.plot_metric_history(
        model_ids=model_ids,
        metric="f1",
        ax=ax
    )
    plt.savefig(join(save_path, f"{string_plot}_f1_metric_history.pdf"), format="pdf", dpi=300)
    plt.show()
    plt.close(fig)

    plotter.plot_confusion_matrix(model_ids=model_ids, save_fig=True, path_save=save_path)
