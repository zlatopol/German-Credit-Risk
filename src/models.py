from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def get_models() -> dict:
    """
    Return baseline models used for credit risk classification.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ),

        "CatBoost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            random_state=42,
            verbose=False
        )
    }

    return models