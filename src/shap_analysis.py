"""SHAP analysis for the final XGBoost credit-risk model.

This module can be run independently with::

    python -m src.shap_analysis

It also exposes reusable helpers for ``shap_report.py`` and other modules.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.metrics import confusion_matrix

from .config import DECISION_THRESHOLD, MODELS_DIR, REPORTS_DIR


MAX_DISPLAY = 15
MODEL_PATH = MODELS_DIR / "xgboost_final.joblib"
TEST_DATA_PATH = REPORTS_DIR / "test_data.csv"
OUTPUT_DIR = REPORTS_DIR / "shap_errors"


# ============================================================
# LOAD DATA
# ============================================================


def load_model_and_test_data(
    model_path: Path = MODEL_PATH,
    test_data_path: Path = TEST_DATA_PATH,
):
    """Load the final XGBoost pipeline and saved test dataset."""

    model_pipeline = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)

    if "Risk" not in test_df.columns:
        raise ValueError("Column 'Risk' was not found in test_data.csv")

    X_test = test_df.drop(columns="Risk")
    y_test = test_df["Risk"]

    return model_pipeline, X_test, y_test


# ============================================================
# PREDICTIONS / ERROR GROUPS
# ============================================================


def get_predictions(
    model_pipeline,
    X_test,
    threshold: float = DECISION_THRESHOLD,
):
    """Generate probabilities and predictions using ``threshold``."""

    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_proba, y_pred


def create_error_groups(y_test, y_pred):
    """Return boolean masks for TP, TN, FP and FN observations."""

    y_true = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    return {
        "TP": (y_true == 1) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
    }


# ============================================================
# MODEL / PREPROCESSING
# ============================================================


def get_model_parts(model_pipeline):
    """Extract the fitted preprocessor and estimator from a pipeline."""

    if not hasattr(model_pipeline, "named_steps"):
        return None, model_pipeline

    steps = model_pipeline.named_steps

    estimator = steps.get("model") or steps.get("classifier")
    if estimator is None:
        estimator = list(steps.values())[-1]

    preprocessor = next(
        (
            steps[name]
            for name in ("preprocessor", "preprocess", "transformer")
            if name in steps
        ),
        None,
    )

    return preprocessor, estimator


def transform_test_data(model_pipeline, X_test):
    """Apply the fitted preprocessing and return transformed data + names."""

    preprocessor, _ = get_model_parts(model_pipeline)

    if preprocessor is None:
        return X_test.to_numpy(), np.asarray(X_test.columns, dtype=object)

    X_transformed = preprocessor.transform(X_test)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_transformed = np.asarray(X_transformed)

    try:
        feature_names = np.asarray(
            preprocessor.get_feature_names_out(),
            dtype=object,
        )
    except Exception:
        feature_names = np.asarray(
            [f"feature_{i}" for i in range(X_transformed.shape[1])],
            dtype=object,
        )

    if len(feature_names) != X_transformed.shape[1]:
        feature_names = np.asarray(
            [f"feature_{i}" for i in range(X_transformed.shape[1])],
            dtype=object,
        )

    return X_transformed, feature_names


# ============================================================
# SHAP VALUES
# ============================================================


def calculate_shap_values(model_pipeline, X_test):
    """Calculate SHAP values for the fitted XGBoost estimator."""

    _, estimator = get_model_parts(model_pipeline)
    X_test_transformed, feature_names = transform_test_data(
        model_pipeline,
        X_test,
    )

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_test_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1] if base_value.size == 2 else base_value[0]

    return (
        explainer,
        shap_values,
        base_value,
        X_test_transformed,
        feature_names,
    )


# ============================================================
# IMPORTANCE
# ============================================================


def calculate_group_importance(shap_values, feature_names, mask):
    """Calculate mean absolute SHAP importance for one group."""

    if mask.sum() == 0:
        return pd.DataFrame(columns=["Feature", "Importance"])

    importance = np.abs(shap_values[mask]).mean(axis=0)

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def calculate_mean_shap(shap_values, feature_names, mask):
    """Calculate mean signed SHAP value for one group."""

    if mask.sum() == 0:
        return pd.Series(0.0, index=feature_names, dtype=float)

    return pd.Series(
        shap_values[mask].mean(axis=0),
        index=feature_names,
        dtype=float,
    )


def calculate_global_importance(shap_values, feature_names):
    """Calculate global mean absolute SHAP importance."""

    return (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# PLOTS
# ============================================================


def save_bar_plot(group_name, importance_df, output_dir, max_display=MAX_DISPLAY):
    """Save a horizontal SHAP importance bar plot."""

    if importance_df.empty:
        return

    values = importance_df.head(max_display).sort_values("Importance")

    plt.figure(figsize=(10, 7))
    plt.barh(values["Feature"], values["Importance"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title(f"SHAP feature importance — {group_name}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{group_name}_bar.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_beeswarm_plot(
    group_name,
    shap_values,
    X_test_transformed,
    feature_names,
    mask,
    output_dir,
    max_display=MAX_DISPLAY,
):
    """Save a SHAP beeswarm plot for one group."""

    if mask.sum() == 0:
        return

    shap.summary_plot(
        shap_values[mask],
        X_test_transformed[mask],
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP distribution — {group_name}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{group_name}_beeswarm.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_waterfall_plot(
    group_name,
    shap_values,
    base_value,
    X_test_transformed,
    feature_names,
    mask,
    y_test,
    y_pred,
    y_proba,
    output_dir,
    max_display=MAX_DISPLAY,
):
    """Save a representative waterfall plot for one error group."""

    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None

    local_shap = shap_values[indices]
    representative_position = np.argmax(np.abs(local_shap).sum(axis=1))
    test_idx = int(indices[representative_position])

    explanation = shap.Explanation(
        values=shap_values[test_idx],
        base_values=base_value,
        data=X_test_transformed[test_idx],
        feature_names=feature_names,
    )

    shap.plots.waterfall(
        explanation,
        max_display=max_display,
        show=False,
    )
    plt.title(
        f"{group_name} — representative observation\n"
        f"test index={test_idx}, true={y_test.iloc[test_idx]}, "
        f"pred={y_pred[test_idx]}, probability={y_proba[test_idx]:.3f}"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{group_name}_waterfall.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return test_idx


# ============================================================
# SAVE RESULTS
# ============================================================


def save_predictions_with_errors(
    X_test,
    y_test,
    y_pred,
    y_proba,
    error_groups,
    output_dir,
):
    """Save test observations with predictions and TP/TN/FP/FN labels."""

    result_df = X_test.copy()
    result_df["true_risk"] = np.asarray(y_test)
    result_df["predicted_risk"] = y_pred
    result_df["risk_probability"] = y_proba
    result_df["error_type"] = "TN"

    for group_name, mask in error_groups.items():
        result_df.loc[mask, "error_type"] = group_name

    result_df.to_csv(
        output_dir / "test_predictions_with_errors.csv",
        index=False,
    )


# ============================================================
# PUBLIC API
# ============================================================


def run_shap_analysis(
    model_pipeline,
    X_test,
    max_display=MAX_DISPLAY,
    output_dir: Path = OUTPUT_DIR,
    threshold: float = DECISION_THRESHOLD,
):
    """Run global SHAP analysis for an already fitted model.

    Returns ``(explainer, shap_values, X_test_transformed, shap_importance)``.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    explainer, shap_values, _, X_test_transformed, feature_names = (
        calculate_shap_values(model_pipeline, X_test)
    )

    shap_importance = calculate_global_importance(
        shap_values,
        feature_names,
    )

    shap_importance.to_csv(
        output_dir / "global_importance.csv",
        index=False,
    )

    return (
        explainer,
        shap_values,
        X_test_transformed,
        shap_importance,
    )


# ============================================================
# CLI
# ============================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SHAP ERROR ANALYSIS")
    print("=" * 60)

    print("\nLoading final XGBoost model...")
    model_pipeline, X_test, y_test = load_model_and_test_data()
    print(f"Test shape: {X_test.shape}")

    print("\nGenerating predictions...")
    y_proba, y_pred = get_predictions(
        model_pipeline,
        X_test,
        threshold=DECISION_THRESHOLD,
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    error_groups = create_error_groups(y_test, y_pred)

    print("\nError groups:")
    for group_name, mask in error_groups.items():
        print(f"{group_name}: {mask.sum()}")

    print("\nCalculating SHAP values...")
    (
        explainer,
        shap_values,
        base_value,
        X_test_transformed,
        feature_names,
    ) = calculate_shap_values(model_pipeline, X_test)

    print("\n" + "=" * 60)
    print("GLOBAL SHAP IMPORTANCE")
    print("=" * 60)

    global_importance = calculate_global_importance(
        shap_values,
        feature_names,
    )

    print(global_importance.head(MAX_DISPLAY).to_string(index=False))
    global_importance.to_csv(
        OUTPUT_DIR / "global_importance.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS BY ERROR GROUP")
    print("=" * 60)

    representative_indices = {}

    for group_name, mask in error_groups.items():
        print("\n" + "-" * 60)
        print(group_name)
        print("-" * 60)
        print(f"Observations: {mask.sum()}")

        if mask.sum() == 0:
            print("No observations in this group.")
            continue

        importance_df = calculate_group_importance(
            shap_values,
            feature_names,
            mask,
        )

        print("\nTop SHAP features:")
        print(importance_df.head(MAX_DISPLAY).to_string(index=False))

        importance_df.to_csv(
            OUTPUT_DIR / f"{group_name}_importance.csv",
            index=False,
        )

        save_bar_plot(
            group_name,
            importance_df,
            OUTPUT_DIR,
            max_display=MAX_DISPLAY,
        )
        save_beeswarm_plot(
            group_name,
            shap_values,
            X_test_transformed,
            feature_names,
            mask,
            OUTPUT_DIR,
            max_display=MAX_DISPLAY,
        )
        representative_indices[group_name] = save_waterfall_plot(
            group_name,
            shap_values,
            base_value,
            X_test_transformed,
            feature_names,
            mask,
            y_test,
            y_pred,
            y_proba,
            OUTPUT_DIR,
            max_display=MAX_DISPLAY,
        )

    save_predictions_with_errors(
        X_test,
        y_test,
        y_pred,
        y_proba,
        error_groups,
        OUTPUT_DIR,
    )

    print("\n" + "=" * 60)
    print("SHAP ERROR ANALYSIS COMPLETED")
    print("=" * 60)
    print(f"\nThreshold: {DECISION_THRESHOLD}")
    print(f"\nSaved results:\nDirectory: {OUTPUT_DIR}/")
    print("\nRepresentative observations:")

    for group_name, index in representative_indices.items():
        if index is not None:
            print(f"{group_name}: test index {index}")


if __name__ == "__main__":
    main()
