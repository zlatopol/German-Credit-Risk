import numpy as np

from src.evaluation import calculate_metrics, find_best_threshold


def test_calculate_metrics_returns_expected_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([0.10, 0.70, 0.80, 0.20])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    assert metrics["Accuracy"] == 0.5
    assert metrics["Precision"] == 0.5
    assert metrics["Recall"] == 0.5
    assert metrics["F1"] == 0.5
    assert 0.0 <= metrics["ROC-AUC"] <= 1.0


def test_find_best_threshold_uses_threshold_array_length():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.10, 0.20, 0.80, 0.90])

    threshold, f1 = find_best_threshold(y_true, y_proba)

    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_find_best_threshold_does_not_use_missing_final_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.10, 0.20, 0.80, 0.90])

    threshold, _ = find_best_threshold(y_true, y_proba)

    # The last precision/recall pair returned by
    # precision_recall_curve has no corresponding threshold.
    assert threshold in {0.10, 0.20, 0.80}
