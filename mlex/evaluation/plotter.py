import matplotlib.pyplot as plt
from os.path import join
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List, Union


class EvaluationPlotter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._load_data()

    def _load_data(self):
        self.df = pq.read_table(self.file_path).to_pandas()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

    def plot_roc_curve(self, model_ids: Union[str, List[str]] = None, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        model_ids = [model_ids] if isinstance(model_ids, str) else model_ids

        ax.plot([0, 1], [0, 1], "k--", linewidth=4, label='random classifier')
        for model_id in model_ids:
            row = self.df[self.df['model_id'] == model_id].squeeze()
            if len(row['fpr']) > 0:
                label = f"{model_id.split('_')[4].capitalize()} context (AUC = {row['auc_roc']:.2f})"
                ax.plot(row['fpr'], row['tpr'], linewidth=4, label=label)

        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=16)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=16)
        ax.set_title(f"Receiver Operating Characteristic\n{model_ids[0].split('_')[0].upper()}", fontsize=18)
        ax.legend(loc="lower right")
        return ax

    def plot_pr_curve(self, model_ids: Union[str, List[str]] = None, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        model_ids = [model_ids] if isinstance(model_ids, str) else model_ids

        y_true = self.df[self.df['model_id'].isin(model_ids)].iloc[0,].squeeze()['y_true']
        no_skill = len(y_true[y_true == 1]) / len(y_true)

        ax.plot([0, 1], [no_skill, no_skill], "k--", linewidth=4, label='random classifier')
        for model_id in model_ids:
            row = self.df[self.df['model_id'] == model_id].squeeze()
            if len(row['rr']) > 0:
                label = f"{model_id.split('_')[4].capitalize()} context (AUC = {row['auc_pr']:.2f})"
                ax.plot(row['rr'], row['pr'], linewidth=4, label=label)

        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("Precision", fontsize=16)
        ax.set_ylabel("Recall", fontsize=16)
        ax.set_title(f"Precision-Recall\n{model_ids[0].split('_')[0].upper()}", fontsize=18)
        ax.legend(loc="lower right")
        return ax

    def plot_confusion_matrix(self, model_ids: List[str], normalize=False, save_fig=False, path_save=None):
        model_ids = [model_ids] if isinstance(model_ids, str) else model_ids

        for model_id in model_ids:
            fig, ax = plt.subplots(figsize=(4, 4))

            model_data = self.df[self.df['model_id'] == model_id].squeeze()
            if model_data.empty:
                raise ValueError(f"No data found for model_id: {model_id}")

            cm = confusion_matrix(
                model_data['y_true'],
                model_data['y_pred']
            )

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            ax.matshow(cm, cmap=plt.get_cmap('Blues'), alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actual', fontsize=18)
            plt.title(f"{' '.join(model_id.split('_')[0:2])} {model_id.split('_')[2].capitalize()} Context", fontsize=16)
            plt.tight_layout()
            if save_fig:
                plt.savefig(join(path_save, f"{'_'.join(model_id.split('_')[0:3])}_confusion_matrix.pdf"), format='pdf', dpi=300)
            plt.show()
            plt.close(fig)

    def plot_metric_history(self, model_ids: List[str], metric: str, ax=None):
        model_ids = [model_ids] if isinstance(model_ids, str) else model_ids

        if not ax:
            fig, ax = plt.subplots(figsize=(10, 6))

        bar_containers = []

        for model_id in model_ids:
            filtered_df = self.df[self.df['model_id'] == model_id].squeeze()
            bars = ax.bar(
                f"{model_id.split('_')[4].capitalize()} context",
                filtered_df[metric],
                label=model_id
            )
            bar_containers.append(bars)

        for bars in bar_containers:
            ax.bar_label(bars, fmt='%.3f', padding=1)

        ax.set_title(f"{metric.upper()} Metric\n{' '.join(model_ids[0].split('_')[0:2])}")
        ax.set_ylabel(f"{metric.upper()} [%]")
        plt.xticks(size=8)
        return ax

    @staticmethod
    def __interpolate_roc_curves(fpr_list, tpr_list, num_points=100):
        common_fpr = np.linspace(0, 1, num_points)
        interpolated_tprs = []
        for fpr, tpr in zip(fpr_list, tpr_list):
            if len(fpr) == 0 or len(tpr) == 0:
                continue
            if fpr[0] > 0:
                fpr = np.insert(fpr, 0, 0.0)
                tpr = np.insert(tpr, 0, 0.0)
            if fpr[-1] < 1.0:
                fpr = np.append(fpr, 1.0)
                tpr = np.append(tpr, tpr[-1])
            interp_tpr = np.interp(common_fpr, fpr, tpr)
            interpolated_tprs.append(interp_tpr)
        return common_fpr, interpolated_tprs

    @staticmethod
    def __interpolate_pr_curves(rr_list, pr_list, num_points=100):
        common_rr = np.linspace(0, 1, num_points)
        interpolated_prs = []
        for rr, pr in zip(rr_list, pr_list):
            if len(rr) == 0 or len(pr) == 0:
                continue
            if rr[0] > 0:
                rr = np.insert(rr, 0, 0.0)
                pr = np.insert(pr, 0, 1.0)
            if rr[-1] < 1.0:
                rr = np.append(rr, 1.0)
                pr = np.append(pr, 0.0)
            interp_pr = np.interp(common_rr, rr, pr)
            interpolated_prs.append(interp_pr)
        return common_rr, interpolated_prs

    def plot_roc_curve_with_ci(self, model_groups: List[List[str]], ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], "k--", linewidth=4, label='Random Classifier')

        for group in model_groups:
            fpr_list, tpr_list, auc_list = [], [], []
            for model_id in group:
                row = self.df[self.df['model_id'] == model_id].squeeze()
                if len(row.get('fpr', [])) > 0:
                    fpr_list.append(row['fpr'])
                    tpr_list.append(row['tpr'])
                    auc_list.append(row['auc_roc'])
            if not fpr_list:
                continue
            common_fpr, interpolated_tprs = self.__interpolate_roc_curves(fpr_list, tpr_list)
            if not interpolated_tprs:
                continue
            mean_tpr = np.mean(interpolated_tprs, axis=0)
            ci_lower = np.percentile(interpolated_tprs, 2.5, axis=0)
            ci_upper = np.percentile(interpolated_tprs, 97.5, axis=0)
            mean_auc = np.mean(auc_list)
            std_auc = np.std(auc_list)
            context = group[0].split('_')[4].capitalize()
            ax.plot(common_fpr, mean_tpr, linewidth=4,
                    label=f"{context} Context (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
            ax.fill_between(common_fpr, ci_lower, ci_upper, alpha=0.3)

        ax.set(xlim=[0, 1], ylim=[0, 1.05],
               xlabel="False Positive Rate",
               ylabel="True Positive Rate",
               title=f"ROC Curve with 95% CI\n{model_groups[0][0].split('_')[0].upper()}")
        ax.legend(loc="lower right")
        return ax

    def plot_pr_curve_with_ci(self, model_groups: List[List[str]], ax=None):
        if not ax:
            fig, ax = plt.subplots()
        if model_groups:
            first_row = self.df[self.df['model_id'] == model_groups[0][0]].squeeze()
            y_true = first_row['y_true']
            no_skill = np.mean(y_true)
            ax.plot([0, 1], [no_skill, no_skill], "k--", label='No Skill')

        for group in model_groups:
            rr_list, pr_list, auc_list = [], [], []
            for model_id in group:
                row = self.df[self.df['model_id'] == model_id].squeeze()
                if len(row.get('rr', [])) > 0:
                    rr_list.append(row['rr'])
                    pr_list.append(row['pr'])
                    auc_list.append(row['auc_pr'])
            if not rr_list:
                continue
            common_rr, interpolated_prs = self.__interpolate_pr_curves(rr_list, pr_list)
            if not interpolated_prs:
                continue
            mean_pr = np.mean(interpolated_prs, axis=0)
            ci_lower = np.percentile(interpolated_prs, 2.5, axis=0)
            ci_upper = np.percentile(interpolated_prs, 97.5, axis=0)
            mean_auc = np.mean(auc_list)
            std_auc = np.std(auc_list)
            context = group[0].split('_')[4].capitalize()
            ax.plot(common_rr, mean_pr, linewidth=4,
                    label=f"{context} Context (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
            ax.fill_between(common_rr, ci_lower, ci_upper, alpha=0.3)

        ax.set(xlim=[0, 1], ylim=[0, 1.05],
               xlabel="Recall",
               ylabel="Precision",
               title=f"PR Curve with 95% CI\n{model_groups[0][0].split('_')[0].upper()}")
        ax.legend(loc="lower right")
        return ax

    # def plot_metric_history_with_ci(self, model_groups: List[List[str]], metric: str, ax=None):
    #     if not ax:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #
    #     group_data = []
    #     labels = []
    #     for group in model_groups:
    #         metric_values = []
    #         for model_id in group:
    #             row = self.df[self.df['model_id'] == model_id].squeeze()
    #             if not row.empty:
    #                 metric_values.append(row[metric])
    #         if metric_values:
    #             mean = np.mean(metric_values)
    #             lower = np.percentile(metric_values, 2.5)
    #             upper = np.percentile(metric_values, 97.5)
    #             group_data.append((mean, lower, upper))
    #             labels.append(f"{group[0].split('_')[4].capitalize()} Context")
    #
    #     x = np.arange(len(labels))
    #     means = [data[0] for data in group_data]
    #     lower_errors = [data[0] - data[1] for data in group_data]
    #     upper_errors = [data[2] - data[0] for data in group_data]
    #     ax.bar(x, means, yerr=[lower_errors, upper_errors], capsize=5, alpha=0.7)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(labels, ha='right')
    #     ax.set_title(f"{metric.upper()} with 95% CI\n{model_groups[0][0].split('_')[0].upper()}")
    #     ax.set_ylabel(metric.upper())
    #     return ax

    def plot_metric_history_with_ci(
            self,
            model_groups: List[List[str]],
            metric: str,
            *,
            ax=None,
            horizontal_cap_width: float = 0.3,
            ci_color: str = "#2187bb",
            marker: str = "o",
            marker_size: float = 6,
            annotate: bool = True,
    ):
        """
        For each group of model_ids, compute the mean ± 95% CI of `metric` and plot
        them as vertical lines with horizontal caps and a mean marker.
        """
        # create axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # extract and compute statistics
        means, lowers, uppers, labels = [], [], [], []
        for group in model_groups:
            vals = [
                float(ser.iloc[0])
                for mid in group
                for ser in [self.df.loc[self.df['model_id'] == mid, metric]]
                if not ser.empty
            ]
            if not vals:
                continue
            mean = np.mean(vals)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            means.append(mean)
            lowers.append(lo)
            uppers.append(hi)
            # label by some group name logic
            labels.append(group[0].split('_')[4].capitalize() + " Context")

        x = np.arange(1, len(means) + 1)

        # plot each interval
        for xi, mean, lo, hi in zip(x, means, lowers, uppers):
            left = xi - horizontal_cap_width / 2
            right = xi + horizontal_cap_width / 2

            # vertical line
            ax.plot([xi, xi], [lo, hi], color=ci_color, linewidth=1.5)
            # caps
            ax.plot([left, right], [lo, lo], color=ci_color, linewidth=1.5)
            ax.plot([left, right], [hi, hi], color=ci_color, linewidth=1.5)
            # mean marker
            ax.plot(xi, mean, marker, color='red', markersize=marker_size)

            # optional annotations
            if annotate:
                ax.text(right + 0.02, hi, f"{hi:.2f}", va="center", ha="left", color=ci_color)
                ax.text(right + 0.02, lo, f"{lo:.2f}", va="center", ha="left", color=ci_color)
                ax.text(xi + 0.03, mean, f"{mean:.2f}", va="center", ha="left", color='red')

        # formatting
        ax.set_xticks(x)
        ax.set_xticklabels(labels, ha="center")
        ax.set_ylim(0, 1)
        title_prefix = model_groups[0][0].split('_')[0].upper()
        ax.set_title(f"{title_prefix}: {metric.upper()} with 95% CI")
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("")  # or something like “Model context”
        ax.margins(x=0.1)  # give a little horizontal padding

        return ax