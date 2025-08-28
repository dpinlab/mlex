import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import matplotlib.pyplot as plt
from mlex.evaluation.plotter import EvaluationPlotter
import pyarrow.parquet as pq


file_path = "evaluation.parquet"
#plotter = EvaluationPlotter("evaluation.parquet")

df = pq.read_table(file_path).to_pandas()

fig, ax = plt.subplots()
model_ids = ["MLP_f1max_f1max"]
if not ax:
    fig, ax = plt.subplots()

model_ids = [model_ids] if isinstance(model_ids, str) else model_ids

ax.plot([0, 1], [0, 1], "k--", linewidth=4, label='random classifier')
for model_id in model_ids:
    row = df[df['model_id'] == model_id].squeeze()
    if len(row['fpr']) > 0:
        label = f"{model_id}"
        ax.plot(row['fpr'], row['tpr'], linewidth=4, label=label)

ax.set_xlim((0.0, 1.0))
ax.set_ylim((0.0, 1.05))
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=16)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=16)
ax.set_title(f"Receiver Operating Characteristic\n{model_ids[0].split('_')[0].upper()}", fontsize=18)
ax.legend(loc="lower right")

plt.savefig("roc_curve_mlp.png", bbox_inches='tight')


plt.show()
plt.close(fig)
plt.clf()


# fig = plotter.plot_metric_history(
#     model_ids=["test_model_1", "test_model_2", "test_model_3"],
#     metric="f1"
# )
# plt.show()
# plt.close(fig)
# plt.clf()
