import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    EXTRATREES_PARAMS,
    FIGURES_DIR,
    LIGHTGBM_PARAMS,
    N_ESTIMATORS_GRID,
    PUMAP_DIR,
    SENSITIVITY_DIR,
    XGBOOST_PARAMS,
)
from utils import ensure_dir, iter_fold_dirs, metrics_dict, scaled_embedding_fold

try:
    import lightgbm as lgb
except Exception as exc:
    raise ImportError("Install lightgbm before running this script.") from exc

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise ImportError("Install xgboost before running this script.") from exc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=Path, default=PUMAP_DIR)
    p.add_argument("--out", type=Path, default=SENSITIVITY_DIR)
    p.add_argument("--figures", type=Path, default=FIGURES_DIR)
    return p.parse_args()


def build_model(model_name, n_estimators):
    if model_name == "ExtraTrees":
        params = {**EXTRATREES_PARAMS, "n_estimators": n_estimators}
        return ExtraTreesClassifier(**params)
    if model_name == "LightGBM":
        params = {**LIGHTGBM_PARAMS, "n_estimators": n_estimators}
        return lgb.LGBMClassifier(**params)
    if model_name == "XGBoost":
        params = {**XGBOOST_PARAMS, "n_estimators": n_estimators}
        return XGBClassifier(**params)
    raise ValueError(model_name)


def fit_model(model, model_name, data):
    if model_name == "LightGBM":
        model.fit(
            data["Z_train"], data["y_train"],
            eval_set=[(data["Z_val"], data["y_val"])],
            eval_metric="multi_logloss",
            callbacks=[lgb.log_evaluation(0)],
        )
    elif model_name == "XGBoost":
        model.fit(
            data["Z_train"], data["y_train"],
            eval_set=[(data["Z_val"], data["y_val"])],
            verbose=False,
        )
    else:
        model.fit(data["Z_train"], data["y_train"])
    return model


def evaluate(model_name, n_estimators, fold_dirs):
    y_true_all, y_pred_all = [], []
    for fold_dir in fold_dirs:
        data, _ = scaled_embedding_fold(fold_dir)
        model = fit_model(build_model(model_name, n_estimators), model_name, data)
        pred = model.predict(data["Z_test"])
        y_true_all.extend(data["y_test"].tolist())
        y_pred_all.extend(pred.tolist())
    return {"model": model_name, "n_estimators": n_estimators, **metrics_dict(np.array(y_true_all), np.array(y_pred_all))}


def plot_results(df, out_path):
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for model_name in ["ExtraTrees", "LightGBM", "XGBoost"]:
        d = df[df["model"] == model_name].sort_values("n_estimators")
        ax.plot(d["n_estimators"], d["accuracy"], marker="o", linewidth=2.5, label=model_name)

    selected = {"ExtraTrees": 650, "LightGBM": 50, "XGBoost": 100}
    for model_name, n_sel in selected.items():
        d = df[(df["model"] == model_name) & (df["n_estimators"] == n_sel)]
        if len(d) == 1:
            ax.scatter(d["n_estimators"], d["accuracy"], s=110, edgecolors="black", zorder=5)

    ax.set_xlabel("Number of estimators", fontsize=16)
    ax.set_ylabel("Global OOF Accuracy", fontsize=16)
    ax.set_title("Tree-based classifier performance versus number of estimators", fontsize=18)
    ax.set_xticks(N_ESTIMATORS_GRID)
    ax.set_ylim(0.972, 0.984)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out = ensure_dir(args.out)
    figures = ensure_dir(args.figures)
    fold_dirs = iter_fold_dirs(args.embeddings)

    rows = []
    for model_name in ["ExtraTrees", "LightGBM", "XGBoost"]:
        for n_estimators in N_ESTIMATORS_GRID:
            print(f"{model_name} n_estimators={n_estimators}")
            rows.append(evaluate(model_name, n_estimators, fold_dirs))

    df = pd.DataFrame(rows)
    df.to_csv(out / "tree_models_accuracy_vs_n_estimators.csv", index=False, encoding="utf-8-sig")
    df.to_csv(figures / "tree_models_accuracy_vs_n_estimators.csv", index=False, encoding="utf-8-sig")
    plot_results(df, figures / "tree_models_accuracy_vs_n_estimators.pdf")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
