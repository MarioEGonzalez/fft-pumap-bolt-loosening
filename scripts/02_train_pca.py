import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import N_COMPONENTS, PCA_DIR, PREPARED_DIR, SEED
from utils import ensure_dir, iter_fold_dirs, read_split, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    p.add_argument("--out", type=Path, default=PCA_DIR)
    p.add_argument("--n-components", type=int, default=N_COMPONENTS)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(args.out)
    data = np.load(args.prepared / "features.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    rows = []
    for fold_dir in iter_fold_dirs(args.prepared):
        train_idx, val_idx, test_idx = read_split(fold_dir)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        X_test = scaler.transform(X[test_idx])

        pca = PCA(n_components=args.n_components, random_state=args.seed)
        Z_train = pca.fit_transform(X_train)
        Z_val = pca.transform(X_val)
        Z_test = pca.transform(X_test)

        out_fold = ensure_dir(out / fold_dir.name)
        np.savez_compressed(
            out_fold / "embeddings_fold.npz",
            Z_train=Z_train,
            y_train=y[train_idx],
            Z_val=Z_val,
            y_val=y[val_idx],
            Z_test=Z_test,
            y_test=y[test_idx],
        )
        dump(scaler, out_fold / "scaler.joblib")
        dump(pca, out_fold / "pca.joblib")

        rows.append({
            "fold": fold_dir.name,
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        })

    save_json({
        "prepared": str(args.prepared),
        "output": str(out),
        "n_components": args.n_components,
        "folds": rows,
    }, out / "metadata.json")
    print(f"PCA embeddings saved to: {out}")


if __name__ == "__main__":
    main()
