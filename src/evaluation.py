import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)


def get_predictions(model, X):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return y_pred, y_proba


def calculate_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def evaluate_model(model, X, y):
    y_pred, y_proba = get_predictions(model, X)

    return calculate_metrics(
        y,
        y_pred,
        y_proba
    )


def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(
        y_true,
        y_proba
    )

    f1_scores = (
        2 * precisions * recalls
        / (precisions + recalls + 1e-10)
    )

    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]

    return best_threshold, f1_scores[best_idx]

def get_feature_importance(pipeline):
    """
    Extract feature importance from a fitted tree-based pipeline.
    """

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    return importance_df