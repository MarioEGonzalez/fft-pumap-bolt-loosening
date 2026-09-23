import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import FIGURES_DIR, PCA_CLASSIFIER_DIR, PUMAP_CLASSIFIER_DIR, SENSITIVITY_DIR
from utils import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pca-results", type=Path, default=PCA_CLASSIFIER_DIR)
    p.add_argument("--pumap-results", type=Path, default=PUMAP_CLASSIFIER_DIR)
    p.add_argument("--sensitivity", type=Path, default=SENSITIVITY_DIR)
    p.add_argument("--figures", type=Path, default=FIGURES_DIR)
    return p.parse_args()


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_vs_pumap(pca_df, pumap_df, figures):
    pca = pca_df.iloc[0]
    pumap = pumap_df[pumap_df["model"] == "ExtraTrees"].iloc[0]
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "mcc"]
    labels = ["Accuracy", "Balanced Acc.", "Macro-F1", "MCC"]

    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6.5))
    b1 = ax.bar(x - width / 2, [pca[m] for m in metrics], width, label="PCA + ExtraTrees")
    b2 = ax.bar(x + width / 2, [pumap[m] for m in metrics], width, label="Parametric UMAP + ExtraTrees")

    ax.set_ylabel("Global OOF score", fontsize=16)
    ax.set_xlabel("Metric", fontsize=16)
    ax.set_title("PCA versus Parametric UMAP using ExtraTrees", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0.965, 0.986)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=12)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.00035, f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    savefig(fig, figures / "pca_vs_pumap_metrics_barplot.pdf")

    pd.DataFrame([
        {"Representation": "PCA", **{m: pca[m] for m in metrics}},
        {"Representation": "Parametric UMAP", **{m: pumap[m] for m in metrics}},
    ]).to_csv(figures / "pca_vs_pumap_metrics_barplot.csv", index=False, encoding="utf-8-sig")


def plot_sensitivity(sensitivity_csv, figures):
    if not sensitivity_csv.exists():
        print(f"Skipping sensitivity figure; missing {sensitivity_csv}")
        return
    df = pd.read_csv(sensitivity_csv)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for model_name in ["ExtraTrees", "LightGBM", "XGBoost"]:
        d = df[df["model"] == model_name].sort_values("n_estimators")
        ax.plot(d["n_estimators"], d["accuracy"], marker="o", linewidth=2.5, label=model_name)
    ax.set_xlabel("Number of estimators", fontsize=16)
    ax.set_ylabel("Global OOF Accuracy", fontsize=16)
    ax.set_title("Tree-based classifier performance versus number of estimators", fontsize=18)
    ax.set_ylim(0.972, 0.984)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    savefig(fig, figures / "tree_models_accuracy_vs_n_estimators.pdf")


def main():
    args = parse_args()
    figures = ensure_dir(args.figures)
    pca_df = pd.read_csv(args.pca_results / "tabla_resumen_modelos_pca.csv")
    pumap_df = pd.read_csv(args.pumap_results / "tabla_resumen_modelos_selected.csv")

    plot_pca_vs_pumap(pca_df, pumap_df, figures)
    plot_sensitivity(args.sensitivity / "tree_models_accuracy_vs_n_estimators.csv", figures)

    table9 = pd.DataFrame([
        {"Representation": "PCA", "Classifier": "ExtraTrees", **pca_df.iloc[0].to_dict()},
        {"Representation": "Parametric UMAP", "Classifier": "ExtraTrees", **pumap_df[pumap_df["model"] == "ExtraTrees"].iloc[0].to_dict()},
    ])
    table9.to_csv(figures / "table9_pca_vs_pumap.csv", index=False, encoding="utf-8-sig")
    pumap_df.to_csv(figures / "table10_classifiers_pumap.csv", index=False, encoding="utf-8-sig")

    print("Generated final figure/table CSV files in:", figures)


if __name__ == "__main__":
    main()
