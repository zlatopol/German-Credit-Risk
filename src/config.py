from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Data
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "german_credit_data.csv"


# Reproducibility
RANDOM_STATE = 42


# Dataset split
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25


# Cross-validation
N_SPLITS = 5


# Model tuning
N_ITER_RANDOM_SEARCH = 30


# Classification
# Selected on the validation set to maximize F1.
DECISION_THRESHOLD = 0.236


# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


XGB_PARAM_GRID = {
    "model__n_estimators": [100, 200, 300, 500, 800],
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5, 10],
    "model__gamma": [0, 0.1, 0.3, 0.5],
}
