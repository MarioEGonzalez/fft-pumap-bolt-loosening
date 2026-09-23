import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import N_COMPONENTS, PREPARED_DIR, PUMAP_DIR, SEED
from utils import ensure_dir, iter_fold_dirs, read_split, save_json

try:
    from umap.parametric_umap import ParametricUMAP
except Exception as exc:
    raise ImportError(
        "ParametricUMAP is required. Install umap-learn with parametric support "
        "and TensorFlow before running this script."
    ) from exc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    p.add_argument("--out", type=Path, default=PUMAP_DIR)
    p.add_argument("--n-components", type=int, default=N_COMPONENTS)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="euclidean")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(args.out)
    data = np.load(args.prepared / "features.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    for fold_dir in iter_fold_dirs(args.prepared):
        train_idx, val_idx, test_idx = read_split(fold_dir)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        X_test = scaler.transform(X[test_idx])

        reducer = ParametricUMAP(
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            random_state=args.seed,
            verbose=True,
        )
        Z_train = reducer.fit_transform(X_train, y=y[train_idx])
        Z_val = reducer.transform(X_val)
        Z_test = reducer.transform(X_test)

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
        reducer.save(str(out_fold / "parametric_umap"))

    save_json(vars(args), out / "metadata.json")
    print(f"Parametric UMAP embeddings saved to: {out}")


if __name__ == "__main__":
    main()
