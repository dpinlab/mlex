import matplotlib.pyplot as plt
from mlex.evaluation.plotter import EvaluationPlotter


plotter = EvaluationPlotter("evaluation.parquet")

fig, ax = plt.subplots()
plotter.plot_roc_curve(["test_model_1", "test_model_2", "test_model_3"], ax=ax)
plt.show()
plt.close(fig)
plt.clf()


fig = plotter.plot_metric_history(
    model_ids=["test_model_1", "test_model_2", "test_model_3"],
    metric="f1"
)
plt.show()
plt.close(fig)
plt.clf()
