import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    EXTRATREES_PARAMS,
    FIGURES_DIR,
    LIGHTGBM_PARAMS,
    MLP_PARAMS,
    PUMAP_CLASSIFIER_DIR,
    PUMAP_DIR,
    XGBOOST_PARAMS,
)
from utils import (
    confusion_counts,
    ensure_dir,
    iter_fold_dirs,
    metrics_dict,
    plot_confusion_matrix,
    save_model_and_scaler,
    scaled_embedding_fold,
)

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
    p.add_argument("--out", type=Path, default=PUMAP_CLASSIFIER_DIR)
    p.add_argument("--figures", type=Path, default=FIGURES_DIR)
    return p.parse_args()


def build_model(name):
    if name == "ExtraTrees":
        return ExtraTreesClassifier(**EXTRATREES_PARAMS)
    if name == "LightGBM":
        return lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
    if name == "XGBoost":
        return XGBClassifier(**XGBOOST_PARAMS)
    if name == "MLP":
        return MLPClassifier(**MLP_PARAMS)
    raise ValueError(name)


def fit_model(model, name, data):
    if name == "LightGBM":
        model.fit(
            data["Z_train"],
            data["y_train"],
            eval_set=[(data["Z_val"], data["y_val"])],
            eval_metric="multi_logloss",
            callbacks=[lgb.log_evaluation(0)],
        )
    elif name == "XGBoost":
        model.fit(
            data["Z_train"],
            data["y_train"],
            eval_set=[(data["Z_val"], data["y_val"])],
            verbose=False,
        )
    else:
        model.fit(data["Z_train"], data["y_train"])
    return model


def run_classifier(name, fold_dirs, out, figures):
    y_true_all, y_pred_all = [], []
    cm_total = None
    fold_rows = []
    model_dir = ensure_dir(out / f"models_{name.lower()}")

    for fold_dir in fold_dirs:
        data, scaler = scaled_embedding_fold(fold_dir)
        model = fit_model(build_model(name), name, data)
        pred = model.predict(data["Z_test"])

        cm = confusion_counts(data["y_test"], pred)
        cm_total = cm if cm_total is None else cm_total + cm
        fold_rows.append({"fold": fold_dir.name, **metrics_dict(data["y_test"], pred)})
        y_true_all.extend(data["y_test"].tolist())
        y_pred_all.extend(pred.tolist())

        fold_out = ensure_dir(model_dir / fold_dir.name)
        if name == "LightGBM":
            model.booster_.save_model(str(fold_out / "model.txt"))
        else:
            save_model_and_scaler(model, scaler, fold_out)
        np.savetxt(fold_out / "cm_raw.csv", cm, delimiter=",", fmt="%d")

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    row = {"model": name, **metrics_dict(y_true_all, y_pred_all)}

    pd.DataFrame(fold_rows).to_csv(model_dir / "per_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([row]).to_csv(model_dir / "global_metrics.csv", index=False, encoding="utf-8-sig")
    np.savetxt(model_dir / "cm_total_raw.csv", cm_total, delimiter=",", fmt="%d")

    plot_confusion_matrix(
        cm_total,
        f"{name} confusion matrix",
        figures / f"confusion_matrix_{name}_raw.pdf",
    )
    return row


def main():
    args = parse_args()
    out = ensure_dir(args.out)
    figures = ensure_dir(args.figures)
    fold_dirs = iter_fold_dirs(args.embeddings)

    rows = [run_classifier(name, fold_dirs, out, figures) for name in ["ExtraTrees", "LightGBM", "XGBoost", "MLP"]]
    summary = pd.DataFrame(rows)[[
        "model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "macro_precision", "macro_recall", "mcc", "kappa", "n_samples"
    ]]
    summary.to_csv(out / "tabla_resumen_modelos_selected.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(figures / "tabla_resumen_modelos_selected.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
