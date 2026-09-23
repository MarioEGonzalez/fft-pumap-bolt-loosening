import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from config import CLASS_NAMES, N_CLASSES


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_npz(path):
    return np.load(path, allow_pickle=True)


def compute_fft_features(x, use_log_magnitude=True, drop_dc=False):
    x = np.asarray(x)
    if x.ndim == 2:
        return x.astype(np.float32)
    if x.ndim != 3:
        raise ValueError("X must be 2D features or 3D windows: (samples, time, channels).")
    spec = np.abs(np.fft.rfft(x, axis=1))
    if drop_dc:
        spec = spec[:, 1:, :]
    if use_log_magnitude:
        spec = np.log1p(spec)
    return spec.reshape(spec.shape[0], -1).astype(np.float32)


def read_split(fold_dir):
    d = load_npz(Path(fold_dir) / "split.npz")
    return d["train_idx"], d["val_idx"], d["test_idx"]


def iter_fold_dirs(root):
    root = Path(root)
    folds = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    if not folds:
        raise FileNotFoundError(f"No fold directories found in {root}")
    return folds


def load_embedding_fold(fold_dir):
    d = load_npz(Path(fold_dir) / "embeddings_fold.npz")
    return {
        "Z_train": d["Z_train"].astype(np.float32),
        "y_train": d["y_train"].astype(np.int64),
        "Z_val": d["Z_val"].astype(np.float32),
        "y_val": d["y_val"].astype(np.int64),
        "Z_test": d["Z_test"].astype(np.float32),
        "y_test": d["y_test"].astype(np.int64),
    }


def scaled_embedding_fold(fold_dir):
    data = load_embedding_fold(fold_dir)
    scaler = StandardScaler()
    data["Z_train"] = scaler.fit_transform(data["Z_train"])
    data["Z_val"] = scaler.transform(data["Z_val"])
    data["Z_test"] = scaler.transform(data["Z_test"])
    return data, scaler


def metrics_dict(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "n_samples": len(y_true),
    }


def row_normalize(cm):
    cm = np.asarray(cm, dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)


def plot_confusion_matrix(cm, title, out_path):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    pct = row_normalize(cm) * 100.0

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(22)
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    ax.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=20,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_and_scaler(model, scaler, out_dir):
    out_dir = ensure_dir(out_dir)
    dump(model, out_dir / "model.joblib")
    dump(scaler, out_dir / "scaler.joblib")


def save_summary(rows, out_path):
    df = pd.DataFrame(rows)
    ensure_dir(Path(out_path).parent)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def confusion_counts(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))
