import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EXTRATREES_PARAMS, FIGURES_DIR, PCA_CLASSIFIER_DIR, PCA_DIR
from utils import (
    confusion_counts,
    ensure_dir,
    iter_fold_dirs,
    metrics_dict,
    plot_confusion_matrix,
    save_model_and_scaler,
    scaled_embedding_fold,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=Path, default=PCA_DIR)
    p.add_argument("--out", type=Path, default=PCA_CLASSIFIER_DIR)
    p.add_argument("--figures", type=Path, default=FIGURES_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(args.out)
    figures = ensure_dir(args.figures)
    model_name = "ExtraTrees"

    y_true_all, y_pred_all = [], []
    cm_total = None
    fold_rows = []

    for fold_dir in iter_fold_dirs(args.embeddings):
        data, scaler = scaled_embedding_fold(fold_dir)
        clf = ExtraTreesClassifier(**EXTRATREES_PARAMS)
        clf.fit(data["Z_train"], data["y_train"])
        pred = clf.predict(data["Z_test"])

        cm = confusion_counts(data["y_test"], pred)
        cm_total = cm if cm_total is None else cm_total + cm
        fold_rows.append({"fold": fold_dir.name, **metrics_dict(data["y_test"], pred)})
        y_true_all.extend(data["y_test"].tolist())
        y_pred_all.extend(pred.tolist())

        save_model_and_scaler(clf, scaler, out / model_name.lower() / fold_dir.name)
        np.savetxt(out / model_name.lower() / fold_dir.name / "cm_raw.csv", cm, delimiter=",", fmt="%d")

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    summary = pd.DataFrame([{"model": model_name, **metrics_dict(y_true_all, y_pred_all)}])
    per_fold = pd.DataFrame(fold_rows)

    per_fold.to_csv(out / "per_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "tabla_resumen_modelos_pca.csv", index=False, encoding="utf-8-sig")
    np.savetxt(out / "cm_total_raw.csv", cm_total, delimiter=",", fmt="%d")

    plot_confusion_matrix(
        cm_total,
        "PCA + ExtraTrees confusion matrix",
        figures / "confusion_matrix_PCA_ExtraTrees_raw.pdf",
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
