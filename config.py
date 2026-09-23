from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

SEED = 42
N_SPLITS = 12
N_COMPONENTS = 32

CLASS_NAMES = ["Healthy", "6Nm", "9Nm", "NoBolt"]
N_CLASSES = len(CLASS_NAMES)

PREPARED_DIR = RESULTS_DIR / "prepared_fft"
PCA_DIR = RESULTS_DIR / "pca"
PUMAP_DIR = RESULTS_DIR / "parametric_umap"
PCA_CLASSIFIER_DIR = RESULTS_DIR / "classifiers_pca"
PUMAP_CLASSIFIER_DIR = RESULTS_DIR / "classifiers_pumap"
SENSITIVITY_DIR = RESULTS_DIR / "sensitivity_n_estimators"

FFT_CONFIG = {
    "use_log_magnitude": True,
    "drop_dc": False,
}

EXTRATREES_PARAMS = {
    "n_estimators": 650,
    "criterion": "gini",
    "max_features": 0.7,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": False,
    "class_weight": "balanced_subsample",
    "max_depth": 30,
    "n_jobs": -1,
    "random_state": SEED,
}

LIGHTGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "learning_rate": 0.05,
    "n_estimators": 50,
    "num_leaves": 127,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
    "verbosity": -1,
}

XGBOOST_PARAMS = {
    "objective": "multi:softmax",
    "num_class": N_CLASSES,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": 4,
    "eval_metric": "mlogloss",
}

MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 256,
    "learning_rate_init": 1e-3,
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.2,
    "n_iter_no_change": 20,
    "random_state": SEED,
}

N_ESTIMATORS_GRID = [50, 100, 200, 300, 400, 500, 650, 800, 1000]
