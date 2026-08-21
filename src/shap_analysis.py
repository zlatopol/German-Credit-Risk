import os

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from .config import RANDOM_STATE


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/xgboost_final.joblib"
TEST_DATA_PATH = "reports/test_data.csv"
OUTPUT_DIR = "reports/shap_errors"

THRESHOLD = 0.236
MAX_DISPLAY = 15


# ============================================================
# LOAD DATA
# ============================================================

def load_model_and_test_data():
    """
    Load the final XGBoost pipeline and test dataset.
    """

    model_pipeline = joblib.load(MODEL_PATH)

    test_df = pd.read_csv(TEST_DATA_PATH)

    if "Risk" not in test_df.columns:
        raise ValueError(
            "Column 'Risk' was not found in test_data.csv"
        )

    X_test = test_df.drop(columns="Risk")
    y_test = test_df["Risk"]

    return model_pipeline, X_test, y_test


# ============================================================
# PREDICTIONS
# ============================================================

def get_predictions(
    model_pipeline,
    X_test,
    threshold=THRESHOLD,
):
    """
    Generate probabilities and predictions.
    """

    y_proba = model_pipeline.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_proba >= threshold
    ).astype(int)

    return y_proba, y_pred


# ============================================================
# ERROR GROUPS
# ============================================================

def create_error_groups(
    y_test,
    y_pred,
):
    """
    Create boolean masks for TP, TN, FP and FN.
    """

    y_true = y_test.to_numpy()

    tp_mask = (
        (y_true == 1)
        & (y_pred == 1)
    )

    tn_mask = (
        (y_true == 0)
        & (y_pred == 0)
    )

    fp_mask = (
        (y_true == 0)
        & (y_pred == 1)
    )

    fn_mask = (
        (y_true == 1)
        & (y_pred == 0)
    )

    return {
        "TP": tp_mask,
        "TN": tn_mask,
        "FP": fp_mask,
        "FN": fn_mask,
    }


# ============================================================
# SHAP VALUES
# ============================================================

def calculate_shap_values(
    model_pipeline,
    X_test,
):
    """
    Calculate SHAP values for the XGBoost model.

    Preprocessing is taken directly from the trained pipeline.
    """

    preprocessor = (
        model_pipeline
        .named_steps["preprocessor"]
    )

    model = (
        model_pipeline
        .named_steps["model"]
    )

    X_test_transformed = (
        preprocessor.transform(X_test)
    )

    # Convert sparse matrix to dense.
    if hasattr(
        X_test_transformed,
        "toarray",
    ):
        X_test_transformed = (
            X_test_transformed.toarray()
        )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_test_transformed
    )

    # XGBoost / SHAP versions can return
    # different formats.
    if isinstance(
        shap_values,
        list,
    ):
        shap_values = shap_values[0]

    base_value = explainer.expected_value

    if isinstance(
        base_value,
        np.ndarray,
    ):
        base_value = base_value[0]

    return (
        explainer,
        shap_values,
        base_value,
        X_test_transformed,
        feature_names,
    )


# ============================================================
# SHAP IMPORTANCE
# ============================================================

def calculate_group_importance(
    shap_values,
    feature_names,
    mask,
):
    """
    Calculate mean absolute SHAP importance
    for one error group.
    """

    if mask.sum() == 0:
        return pd.DataFrame(
            columns=[
                "Feature",
                "Importance",
            ]
        )

    group_shap = shap_values[mask]

    importance = (
        np.abs(group_shap)
        .mean(axis=0)
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance,
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance_df


# ============================================================
# BAR PLOT
# ============================================================

def save_bar_plot(
    group_name,
    importance_df,
    output_dir,
):
    """
    Save SHAP bar plot for one group.
    """

    if importance_df.empty:
        return

    values = (
        importance_df
        .head(MAX_DISPLAY)
        .sort_values(
            "Importance"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        values["Feature"],
        values["Importance"],
    )

    plt.xlabel(
        "Mean |SHAP value|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"SHAP feature importance — {group_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{group_name}_bar.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# BEESWARM
# ============================================================

def save_beeswarm_plot(
    group_name,
    shap_values,
    X_test_transformed,
    feature_names,
    mask,
    output_dir,
):
    """
    Save SHAP beeswarm plot for one group.
    """

    if mask.sum() == 0:
        return

    group_shap = (
        shap_values[mask]
    )

    group_data = (
        X_test_transformed[mask]
    )

    plt.figure(
        figsize=(10, 7)
    )

    shap.summary_plot(
        group_shap,
        group_data,
        feature_names=feature_names,
        max_display=MAX_DISPLAY,
        show=False,
    )

    plt.title(
        f"SHAP distribution — {group_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{group_name}_beeswarm.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# WATERFALL
# ============================================================

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
):
    """
    Save a representative SHAP waterfall plot
    for one TP/TN/FP/FN group.

    Representative observation is selected as
    the observation with the largest total
    absolute SHAP contribution.
    """

    indices = np.where(mask)[0]

    if len(indices) == 0:
        return

    group_shap = (
        shap_values[indices]
    )

    representative_position = np.argmax(
        np.abs(group_shap)
        .sum(axis=1)
    )

    test_idx = (
        indices[
            representative_position
        ]
    )

    explanation = shap.Explanation(
        values=shap_values[test_idx],
        base_values=base_value,
        data=X_test_transformed[test_idx],
        feature_names=feature_names,
    )

    plt.figure(
        figsize=(10, 8)
    )

    shap.plots.waterfall(
        explanation,
        max_display=MAX_DISPLAY,
        show=False,
    )

    plt.title(
        f"{group_name} — representative observation\n"
        f"test index={test_idx}, "
        f"true={y_test.iloc[test_idx]}, "
        f"pred={y_pred[test_idx]}, "
        f"probability={y_proba[test_idx]:.3f}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{group_name}_waterfall.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return test_idx


# ============================================================
# SAVE GROUP IMPORTANCE
# ============================================================

def save_group_importance(
    group_name,
    importance_df,
    output_dir,
):
    """
    Save SHAP importance table.
    """

    importance_df.to_csv(
        os.path.join(
            output_dir,
            f"{group_name}_importance.csv",
        ),
        index=False,
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

def save_predictions_with_errors(
    X_test,
    y_test,
    y_pred,
    y_proba,
    error_groups,
    output_dir,
):
    """
    Save test observations with prediction
    and error-group information.
    """

    result_df = X_test.copy()

    result_df["true_risk"] = (
        y_test.to_numpy()
    )

    result_df["predicted_risk"] = (
        y_pred
    )

    result_df["risk_probability"] = (
        y_proba
    )

    result_df["error_type"] = "TN"

    result_df.loc[
        error_groups["TP"],
        "error_type",
    ] = "TP"

    result_df.loc[
        error_groups["TN"],
        "error_type",
    ] = "TN"

    result_df.loc[
        error_groups["FP"],
        "error_type",
    ] = "FP"

    result_df.loc[
        error_groups["FN"],
        "error_type",
    ] = "FN"

    result_df.to_csv(
        os.path.join(
            output_dir,
            "test_predictions_with_errors.csv",
        ),
        index=False,
    )


# ============================================================
# MAIN SHAP ANALYSIS
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load model and test data
    # --------------------------------------------------------

    print("=" * 60)
    print("SHAP ERROR ANALYSIS")
    print("=" * 60)

    print(
        "\nLoading final XGBoost model..."
    )

    model_pipeline, X_test, y_test = (
        load_model_and_test_data()
    )

    print(
        f"Test shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # 2. Predictions
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    y_proba, y_pred = get_predictions(
        model_pipeline,
        X_test,
        threshold=THRESHOLD,
    )

    # --------------------------------------------------------
    # 3. Confusion matrix
    # --------------------------------------------------------

    print(
        "\nConfusion matrix:"
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    print(cm)

    # --------------------------------------------------------
    # 4. Error groups
    # --------------------------------------------------------

    error_groups = create_error_groups(
        y_test,
        y_pred,
    )

    print(
        "\nError groups:"
    )

    for group_name, mask in (
        error_groups.items()
    ):
        print(
            f"{group_name}: "
            f"{mask.sum()}"
        )

    # --------------------------------------------------------
    # 5. Calculate SHAP
    # --------------------------------------------------------

    print(
        "\nCalculating SHAP values..."
    )

    (
        explainer,
        shap_values,
        base_value,
        X_test_transformed,
        feature_names,
    ) = calculate_shap_values(
        model_pipeline,
        X_test,
    )

    # --------------------------------------------------------
    # 6. Global SHAP importance
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )
    print(
        "GLOBAL SHAP IMPORTANCE"
    )
    print(
        "=" * 60
    )

    global_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": (
            np.abs(shap_values)
            .mean(axis=0)
        ),
    })

    global_importance = (
        global_importance
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        global_importance
        .head(MAX_DISPLAY)
        .to_string(index=False)
    )

    global_importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "global_importance.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # 7. SHAP analysis for TP/TN/FP/FN
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )
    print(
        "SHAP ANALYSIS BY ERROR GROUP"
    )
    print(
        "=" * 60
    )

    representative_indices = {}

    for group_name, mask in (
        error_groups.items()
    ):

        print(
            "\n" + "-" * 60
        )

        print(
            f"{group_name}"
        )

        print(
            "-" * 60
        )

        count = mask.sum()

        print(
            f"Observations: {count}"
        )

        if count == 0:
            print(
                "No observations in this group."
            )
            continue

        # ----------------------------------------------------
        # Importance
        # ----------------------------------------------------

        importance_df = (
            calculate_group_importance(
                shap_values,
                feature_names,
                mask,
            )
        )

        print(
            "\nTop SHAP features:"
        )

        print(
            importance_df
            .head(MAX_DISPLAY)
            .to_string(index=False)
        )

        save_group_importance(
            group_name,
            importance_df,
            OUTPUT_DIR,
        )

        # ----------------------------------------------------
        # Bar plot
        # ----------------------------------------------------

        save_bar_plot(
            group_name,
            importance_df,
            OUTPUT_DIR,
        )

        # ----------------------------------------------------
        # Beeswarm
        # ----------------------------------------------------

        save_beeswarm_plot(
            group_name,
            shap_values,
            X_test_transformed,
            feature_names,
            mask,
            OUTPUT_DIR,
        )

        # ----------------------------------------------------
        # Representative waterfall
        # ----------------------------------------------------

        representative_idx = (
            save_waterfall_plot(
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
            )
        )

        representative_indices[
            group_name
        ] = representative_idx

    # --------------------------------------------------------
    # 8. Save predictions
    # --------------------------------------------------------

    save_predictions_with_errors(
        X_test,
        y_test,
        y_pred,
        y_proba,
        error_groups,
        OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )
    print(
        "SHAP ERROR ANALYSIS COMPLETED"
    )
    print(
        "=" * 60
    )

    print(
        f"\nThreshold: {THRESHOLD}"
    )

    print(
        "\nSaved results:"
    )

    print(
        f"Directory: {OUTPUT_DIR}/"
    )

    print(
        "\nRepresentative observations:"
    )

    for group_name, index in (
        representative_indices.items()
    ):

        if index is not None:
            print(
                f"{group_name}: "
                f"test index {index}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()