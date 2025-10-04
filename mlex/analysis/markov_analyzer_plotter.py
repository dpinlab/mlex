import matplotlib.pyplot as plt
import seaborn as sns
from mlex.utils.utils import ensure_directory_exists


class MarkovAnalyzerPlotter:
    def __init__(self):
        pass

    def plot_markov_chain(self, probability_matrix, column_name, path_save):
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            probability_matrix,
            annot=True,   
            fmt=".5f",
            cmap="Blues",  
            cbar=True,
            annot_kws={"size": 6}
        )

        plt.title(f"Heatmap - Matriz de Probabilidade ({column_name.upper()})")
        plt.xlabel("Próximo estado")
        plt.ylabel("Estado atual")

        plt.tight_layout()
        ensure_directory_exists(path_save)
        plt.savefig(f"{path_save}/heatmap_prob_{column_name.lower()}.png", dpi=300)
        plt.close()
