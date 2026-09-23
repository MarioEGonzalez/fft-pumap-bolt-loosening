import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import FFT_CONFIG, N_SPLITS, PREPARED_DIR, SEED
from utils import compute_fft_features, ensure_dir, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="NPZ with X, y and groups arrays.")
    p.add_argument("--out", type=Path, default=PREPARED_DIR)
    p.add_argument("--n-splits", type=int, default=N_SPLITS)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def make_inner_validation(X, y, groups, train_idx, seed):
    y_tr = y[train_idx]
    g_tr = groups[train_idx]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_train_local, val_local = next(inner.split(X[train_idx], y_tr, g_tr))
    return train_idx[inner_train_local], train_idx[val_local]


def main():
    args = parse_args()
    out = ensure_dir(args.out)

    data = np.load(args.input, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(np.int64)
    groups = data["groups"].astype(str)

    features = compute_fft_features(X, **FFT_CONFIG)
    np.savez_compressed(out / "features.npz", X=features, y=y, groups=groups)

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_rows = []

    for fold, (train_val_idx, test_idx) in enumerate(splitter.split(features, y, groups)):
        train_idx, val_idx = make_inner_validation(features, y, groups, train_val_idx, args.seed + fold)
        fold_dir = ensure_dir(out / f"fold_{fold:02d}")
        np.savez_compressed(
            fold_dir / "split.npz",
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        })

    save_json({
        "input": str(args.input),
        "output": str(out),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "feature_shape": list(features.shape),
        "folds": fold_rows,
    }, out / "metadata.json")

    print(f"Prepared features: {features.shape}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
