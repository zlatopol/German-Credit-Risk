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
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def evaluate_model(model, X, y):
    y_pred, y_proba = get_predictions(model, X)

    return calculate_metrics(
        y,
        y_pred,
        y_proba,
    )


def find_best_threshold(y_true, y_proba):
    """
    Find the decision threshold that maximizes F1.

    precision_recall_curve returns one more precision/recall value
    than thresholds, so the last F1 value has no corresponding
    threshold and must not be used for threshold selection.
    """

    precisions, recalls, thresholds = precision_recall_curve(
        y_true,
        y_proba,
    )

    f1_scores = (
        2 * precisions * recalls
        / (precisions + recalls + 1e-10)
    )

    # The final precision/recall pair has no matching threshold.
    f1_for_thresholds = f1_scores[:-1]

    best_idx = np.argmax(f1_for_thresholds)
    best_threshold = thresholds[best_idx]

    return best_threshold, f1_for_thresholds[best_idx]


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
        "Importance": importance,
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False,
    )

    return importance_df
