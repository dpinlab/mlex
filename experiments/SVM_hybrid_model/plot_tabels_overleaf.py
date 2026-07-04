import re
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ["ai4i", "ozone", "occupancy", "eye-detection", "cobot"]
SEQUENCE_LENGTHS = [10, 20, 30, 40, 50]
HIDDEN_SIZES = [10, 32]
NUM_LAYERS_OPTIONS = [1, 2]
SVM_CONFIGS = [
    (0.1, "scale"),
    (0.1, "auto"),
    (1.0, "scale"),
    (1.0, "auto"),
    (10.0, "scale"),
    (10.0, "auto"),
]
OUTPUT_DIR = Path(__file__).resolve().parent / "overleaf_tables"
OUTPUT_DIR.mkdir(exist_ok=True)


def _interpolate_roc_curves(fpr_list, tpr_list, num_points=200):
    common_fpr = np.linspace(0.0, 1.0, num_points)
    interpolated_tprs = []
    for fpr, tpr in zip(fpr_list, tpr_list):
        if len(fpr) == 0 or len(tpr) == 0:
            continue
        fpr = np.asarray(fpr, dtype=float)
        tpr = np.asarray(tpr, dtype=float)
        if fpr[0] > 0:
            fpr = np.insert(fpr, 0, 0.0)
            tpr = np.insert(tpr, 0, 0.0)
        if fpr[-1] < 1.0:
            fpr = np.append(fpr, 1.0)
            tpr = np.append(tpr, tpr[-1])
        interpolated_tprs.append(np.interp(common_fpr, fpr, tpr))
    return common_fpr, interpolated_tprs


def _auc_from_interpolated_curves(fpr_list, tpr_list):
    if not fpr_list or not tpr_list:
        return np.nan
    common_fpr, interpolated_tprs = _interpolate_roc_curves(fpr_list, tpr_list)
    if not interpolated_tprs:
        return np.nan
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    return float(np.trapz(mean_tpr, common_fpr))


def _parse_model_id(model_id):
    model_id = str(model_id).strip()
    pattern = re.compile(
        r"^(?P<dataset>.+?)-(?P<kind>pure-(?P<arch>rnn|lstm|gru)|hybrid)"
        r"_Layers-(?P<num_layers>\d+)"
        r"_HiddenSize-(?P<hidden_size>\d+)"
        r"_SequenceLength-(?P<seq_len>\d+)"
        r"(?:_Iteration-(?P<iteration>\d+))?"
        r"(?:_svmC-(?P<svm_c>[-+]?\d*\.?\d+))?"
        r"(?:_svmGamma-(?P<svm_gamma>scale|auto))?$"
    )
    match = pattern.match(model_id)
    if not match:
        return None
    data = match.groupdict()
    if data["kind"].startswith("pure-"):
        data["kind"] = data["kind"].split("-", 1)[1]
    data["num_layers"] = int(data["num_layers"])
    data["hidden_size"] = int(data["hidden_size"])
    data["seq_len"] = int(data["seq_len"])
    if data.get("svm_c") is not None:
        data["svm_c"] = float(data["svm_c"])
    return data


def _summarize_group(rows_df):
    fpr_list = []
    tpr_list = []
    auc_list = []
    for _, row in rows_df.iterrows():
        if isinstance(row.get("fpr"), (list, tuple, np.ndarray)):
            fpr_list.append(np.asarray(row["fpr"], dtype=float))
        if isinstance(row.get("tpr"), (list, tuple, np.ndarray)):
            tpr_list.append(np.asarray(row["tpr"], dtype=float))
        if pd.notna(row.get("auc_roc")):
            auc_list.append(float(row["auc_roc"]))

    if fpr_list and tpr_list:
        auc_value = _auc_from_interpolated_curves(fpr_list, tpr_list)
        if np.isfinite(auc_value):
            return auc_value, float(np.std(auc_list)) if auc_list else 0.0

    if auc_list:
        mean_auc = float(np.mean(auc_list))
        std_auc = float(np.std(auc_list)) if len(auc_list) > 1 else 0.0
        return mean_auc, std_auc

    return np.nan, np.nan


def _format_cell(value, best_value=None):
    if np.isnan(value):
        return r"---"
    text = rf"${value:.3f}$"
    if best_value is not None and not np.isnan(best_value) and np.isclose(value, best_value):
        return rf"\textbf{{{text}}}"
    return text


def _build_latex_table(dataset_name, df):
    grouped = {}
    for _, row in df.iterrows():
        parsed = _parse_model_id(row["model_id"])
        if not parsed:
            continue
        if parsed["kind"] in {"rnn", "lstm", "gru"}:
            key = (parsed["kind"], parsed["num_layers"], parsed["hidden_size"], parsed["seq_len"])
        else:
            svm_c = parsed.get("svm_c", 1.0)
            svm_gamma = parsed.get("svm_gamma", "scale")
            key = ("hybrid", parsed["num_layers"], parsed["hidden_size"], parsed["seq_len"], svm_c, svm_gamma)
        grouped.setdefault(key, []).append(row)

    summaries = {}
    for key, rows in grouped.items():
        rows_df = pd.DataFrame(rows)
        mean_auc, _ = _summarize_group(rows_df)
        summaries[key] = (mean_auc, _)

    rows = []
    for num_layers in NUM_LAYERS_OPTIONS:
        for hidden_size in HIDDEN_SIZES:
            for seq_len in SEQUENCE_LENGTHS:
                pure_values = []
                for arch in ["rnn", "lstm", "gru"]:
                    key = (arch, num_layers, hidden_size, seq_len)
                    mean_auc, std_auc = summaries.get(key, (np.nan, np.nan))
                    pure_values.append((mean_auc, std_auc))

                hybrid_values = []
                for svm_c, svm_gamma in SVM_CONFIGS:
                    key = ("hybrid", num_layers, hidden_size, seq_len, svm_c, svm_gamma)
                    mean_auc, std_auc = summaries.get(key, (np.nan, np.nan))
                    hybrid_values.append((mean_auc, std_auc))

                rows.append({
                    "layers": num_layers,
                    "hidden_size": hidden_size,
                    "seq_len": seq_len,
                    "pure_values": pure_values,
                    "hybrid_values": hybrid_values,
                })

    all_values = [value for row in rows for value in row["pure_values"] + row["hybrid_values"]]
    best_value = max([v[0] for v in all_values if np.isfinite(v[0])], default=np.nan)

    lines = []
    lines.append(r"\documentclass{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs,multirow,array,graphicx}")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{table}[!t]")
    lines.append(r"  \centering")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{2pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.05}")
    lines.append(rf"  \caption{{AUC-ROC results for the {dataset_name.upper()} dataset. Values are reported as mean $\pm$ std across repetitions and are computed from the same interpolated ROC curves used in the plotting routine.}}")
    lines.append(rf"  \label{{tab:auc_roc_{dataset_name.replace('-', '_')}}}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(r"  \begin{tabular}{c c c c c c c c c c c c}")
    lines.append(r"    \toprule")
    lines.append(r"    \multirow{2}{*}{Layers} & \multirow{2}{*}{Hidden} & \multirow{2}{*}{Seq.} & \multicolumn{3}{c}{Baselines} & \multicolumn{6}{c}{Hybrid RNN-SVM} \\")
    lines.append(r"    \cmidrule(lr){4-6} \cmidrule(lr){7-12}")
    lines.append(r"    & & & RNN & LSTM & GRU & $C=0.1,\gamma=\mathrm{scale}$ & $C=0.1,\gamma=\mathrm{auto}$ & $C=1.0,\gamma=\mathrm{scale}$ & $C=1.0,\gamma=\mathrm{auto}$ & $C=10.0,\gamma=\mathrm{scale}$ & $C=10.0,\gamma=\mathrm{auto}$ \\")
    lines.append(r"    \midrule")

    for num_layers in NUM_LAYERS_OPTIONS:
        for hidden_size in HIDDEN_SIZES:
            lines.append(r"    \cmidrule(lr){1-3}")
            for idx, seq_len in enumerate(SEQUENCE_LENGTHS):
                cells = []
                if idx == 0:
                    cells.append(r"\multirow{5}{*}{" + str(num_layers) + r"}")
                    cells.append(r"\multirow{5}{*}{" + str(hidden_size) + r"}")
                else:
                    cells.append("")
                    cells.append("")
                cells.append(str(seq_len))
                for row in rows:
                    if row["layers"] == num_layers and row["hidden_size"] == hidden_size and row["seq_len"] == seq_len:
                        for mean_auc, _ in row["pure_values"]:
                            cells.append(_format_cell(mean_auc, best_value))
                        for mean_auc, _ in row["hybrid_values"]:
                            cells.append(_format_cell(mean_auc, best_value))
                        break
                lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def main():
    for dataset_name in DATASETS:
        parquet_path = Path(f"{dataset_name}-all_experiments_results.parquet")
        if not parquet_path.exists():
            alt_path = Path(dataset_name) / parquet_path.name
            if alt_path.exists():
                parquet_path = alt_path
            else:
                print(f"Skipping {dataset_name}: parquet not found")
                continue

        df = pd.read_parquet(parquet_path)
        if "model_id" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "model_id"})
        df["model_id"] = df["model_id"].astype(str).str.strip()

        tex = _build_latex_table(dataset_name, df)
        output_path = OUTPUT_DIR / f"{dataset_name}_results_table.tex"
        output_path.write_text(tex, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
