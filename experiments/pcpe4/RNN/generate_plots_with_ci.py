import os
import sys

module_path = os.path.abspath(os.path.join('../../../'))
if module_path not in sys.path:
    sys.path.append(module_path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
from os.path import join
import matplotlib.pyplot as plt
from mlex.evaluation.plotter import EvaluationPlotter
from itertools import product


def ensure_directory_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


models    = ['RNN']
lengths   = ['10', '20', '30', '40', '50']
thresholds_list = ['f1max']
iterations = 10
iteration = 10
num_layers = 1
hidden_size = 10

sequences_compositions = ['temporal', 'Feature_individual', 'Feature_account']

save_path = join("results", f"{num_layers}-layer")
ensure_directory_exists(save_path)
plotter = EvaluationPlotter(f"evaluation.parquet")

# Generate the strings
formatted_ids = [
    f"{model}_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{length}_{seq_comp}_{threshold}_Iteration-{i+1}"
    for model, length, threshold, i, seq_comp in product(
        models, lengths, thresholds_list, range(iterations), sequences_compositions
    )
]


def get_label(group_id: str):
    """
    Gera o label procurando as chaves de composição (temporal, Feature_...) 
    na string do ID (mais robusto que usar o índice fixo parts[4]).
    """
    label_mapping = {
        'temporal': 'Temporal Context',
        'Feature_individual': 'Feature (Individual) Context',
        'Feature_account': 'Feature (Account) Context',
    }
    
    if 'Feature_individual' in group_id:
        return label_mapping['Feature_individual']
    elif 'Feature_account' in group_id:
        return label_mapping['Feature_account']
    elif 'temporal' in group_id:
        return label_mapping['temporal']
  
    if 'Feature' in group_id:
        return "Feature Context (Old ID Format)" 
        
    return "Unknown Context"

for sequence_length in lengths:
    model_ids_for_length = [
        [mid for mid in formatted_ids 
         if f"SequenceLength-{sequence_length}_" in mid and f"_{seq_comp}_" in mid]
        for seq_comp in sequences_compositions
    ]

    # remove grupos vazios
    model_ids_for_length = [g for g in model_ids_for_length if g]

    if not model_ids_for_length:  
        continue

    string_plot = f"seq-{sequence_length}"

    fig, ax = plt.subplots()
    labels = [get_label(group[0]) for group in model_ids_for_length]
    plotter.plot_roc_curve_with_ci(model_ids_for_length, ax=ax, labels=labels,shade=False)
    plt.savefig(join(save_path, f"{string_plot}_eye_detection-roc_curve_with_ci.pdf"), format="pdf", dpi=300)
    plt.show()
    plt.close(fig)

    # fig, ax = plt.subplots()
    # plotter.plot_pr_curve_with_ci(model_ids, ax=ax)
    # plt.savefig(join(save_path, f"{string_plot}_pr_curve_with_ci.pdf"), format="pdf", dpi=300)
    # plt.show()
    # plt.close(fig)

    fig, ax = plt.subplots()
    plotter.plot_metric_history_with_ci(
        model_groups=model_ids_for_length,
        metric="f1",
        ax=ax
    )
    plt.savefig(join(save_path, f"{string_plot}_f1_metric_history_with_ci.pdf"), format="pdf", dpi=300)
    plt.show()
    plt.close(fig)
