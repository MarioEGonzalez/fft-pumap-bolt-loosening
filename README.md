# Bolt-loosening SHM classification pipeline

Clean scripts for the FFT, PCA, Parametric UMAP and supervised-classifier workflow used in the bolt-loosening classification experiments.

## Repository layout

```text
.
├── config.py
├── utils.py
├── scripts/
│   ├── 01_prepare_folds_fft.py
│   ├── 02_train_pca.py
│   ├── 03_train_parametric_umap.py
│   ├── 04_train_classifiers_pca.py
│   ├── 05_train_classifiers_pumap.py
│   ├── 06_generate_figures_tables.py
│   └── 07_sensitivity_n_estimators.py
├── data/
├── results/
├── figures/
└── models/
```

## Input format

The preparation script expects a `.npz` file with:

- `X`: either raw windows with shape `(n_samples, n_time, n_channels)` or precomputed features with shape `(n_samples, n_features)`.
- `y`: integer labels.
- `groups`: experiment/group identifiers used by `StratifiedGroupKFold`.

## Execution order

```bash
python scripts/01_prepare_folds_fft.py --input data/input_windows.npz
python scripts/02_train_pca.py
python scripts/03_train_parametric_umap.py
python scripts/04_train_classifiers_pca.py
python scripts/07_sensitivity_n_estimators.py
python scripts/05_train_classifiers_pumap.py
python scripts/06_generate_figures_tables.py
```

## Final selected classifier settings

- ExtraTrees: `n_estimators=650`
- LightGBM: `n_estimators=50`
- XGBoost: `n_estimators=100`
- MLP: hidden layers `(128, 64)`

## Main outputs

- `figures/pca_vs_pumap_metrics_barplot.pdf`
- `figures/tree_models_accuracy_vs_n_estimators.pdf`
- `figures/confusion_matrix_ExtraTrees_raw.pdf`
- `figures/confusion_matrix_LightGBM_raw.pdf`
- `figures/confusion_matrix_XGBoost_raw.pdf`
- `figures/confusion_matrix_MLP_raw.pdf`
- `figures/table9_pca_vs_pumap.csv`
- `figures/table10_classifiers_pumap.csv`
