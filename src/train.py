import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from .config import (
    DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    XGB_PARAM_GRID,
)

from .data import (
    load_data,
    prepare_target,
)

from .features import create_features
from .preprocessing import build_preprocessor
from .models import get_models

from .evaluation import (
    evaluate_model,
    find_best_threshold,
    get_feature_importance,
)

from .cv import (
    create_cv,
    cross_validate_models,
)


# ============================================================
# DATA SPLIT
# ============================================================

def split_data(df: pd.DataFrame):
    """
    Split data into train, validation and test sets.

    60% train
    20% validation
    20% test

    Stratification preserves the target class distribution.
    """

    X = df.drop(columns="Risk")
    y = df["Risk"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(model):
    """
    Build preprocessing + model pipeline.

    All preprocessing steps, including imputation,
    encoding and scaling, are fitted only on training data.
    """

    preprocessor = build_preprocessor()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_proba,
):
    """
    Calculate classification metrics.
    """

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            y_proba,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. Load data
    # ========================================================

    df = load_data(DATA_PATH)

    print(f"Dataset shape: {df.shape}")


    # ========================================================
    # 2. Prepare target
    # ========================================================

    df = prepare_target(df)

    print("\nTarget distribution:")
    print(
        df["Risk"].value_counts()
    )

    print("\nTarget distribution (%):")
    print(
        df["Risk"].value_counts(
            normalize=True
        )
    )


    # ========================================================
    # 3. Feature engineering
    # ========================================================

    df = create_features(df)


    # ========================================================
    # 4. Train / validation / test split
    # ========================================================

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")


    # ========================================================
    # 5. Baseline models
    # ========================================================

    models = get_models()

    fitted_models = {}
    baseline_results = []

    print("\n" + "=" * 60)
    print("BASELINE MODELS")
    print("=" * 60)

    for name, model in models.items():

        print(f"\nTraining: {name}")

        pipeline = build_pipeline(model)

        pipeline.fit(
            X_train,
            y_train,
        )

        fitted_models[name] = pipeline

        metrics = evaluate_model(
            pipeline,
            X_val,
            y_val,
        )

        metrics["Model"] = name

        baseline_results.append(metrics)


    # ========================================================
    # 6. Baseline comparison
    # ========================================================

    baseline_df = pd.DataFrame(
        baseline_results
    )

    baseline_df = baseline_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ]
    ]

    print("\nValidation results:")
    print(
        baseline_df.to_string(
            index=False
        )
    )


    # ========================================================
    # 7. Cross-validation
    # ========================================================

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)

    cv = create_cv(
        n_splits=5,
        random_state=RANDOM_STATE,
    )

    cv_results = cross_validate_models(
        models=models,
        build_pipeline=build_pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
    )

    print("\nCross-validation results:")
    print(
        cv_results.to_string(
            index=False
        )
    )


    # ========================================================
    # 8. Hyperparameter tuning
    # ========================================================

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING: XGBOOST")
    print("=" * 60)

    xgb_pipeline = build_pipeline(
        models["XGBoost"]
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=XGB_PARAM_GRID,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    random_search.fit(
        X_train,
        y_train,
    )

    print("\nBest XGBoost parameters:")
    print(
        random_search.best_params_
    )

    print(
        f"\nBest CV ROC-AUC: "
        f"{random_search.best_score_:.3f}"
    )


    # ========================================================
    # 9. Threshold tuning
    # ========================================================

    print("\n" + "=" * 60)
    print("THRESHOLD TUNING")
    print("=" * 60)

    tuned_xgb = random_search.best_estimator_

    # Validation data has never been used
    # to fit the tuned model.
    y_val_proba = tuned_xgb.predict_proba(
        X_val
    )[:, 1]

    best_threshold, best_f1 = find_best_threshold(
        y_val,
        y_val_proba,
    )

    print(
        f"\nBest threshold: "
        f"{best_threshold:.3f}"
    )

    print(
        f"Validation F1: "
        f"{best_f1:.3f}"
    )


    # ========================================================
    # 10. Tuned model validation
    # ========================================================

    y_val_pred = (
        y_val_proba >= best_threshold
    ).astype(int)

    validation_metrics = calculate_metrics(
        y_val,
        y_val_pred,
        y_val_proba,
    )

    print("\nTuned XGBoost validation metrics:")

    for metric, value in validation_metrics.items():
        print(
            f"{metric}: "
            f"{value:.3f}"
        )


    # ========================================================
    # 11. Final model
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL MODEL")
    print("=" * 60)

    # After model selection, hyperparameter tuning
    # and threshold tuning, combine train + validation.
    X_train_final = pd.concat(
        [
            X_train,
            X_val,
        ],
        axis=0,
    )

    y_train_final = pd.concat(
        [
            y_train,
            y_val,
        ],
        axis=0,
    )

    final_model = build_pipeline(
        models["XGBoost"]
    )

    final_model.set_params(
        **random_search.best_params_
    )

    print(
        "\nTraining final XGBoost "
        "on train + validation..."
    )

    final_model.fit(
        X_train_final,
        y_train_final,
    )


    # ========================================================
    # 12. Final test evaluation
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    # Test set is used only once.
    # Threshold was selected on validation
    # and is NOT optimized again.

    y_test_proba = final_model.predict_proba(
        X_test
    )[:, 1]

    y_test_pred = (
        y_test_proba >= best_threshold
    ).astype(int)


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_test_pred,
        cmap="Blues",
    )

    plt.title(
        "XGBoost — Confusion Matrix"
    )

    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------

    RocCurveDisplay.from_predictions(
        y_test,
        y_test_proba,
    )

    plt.title(
        "XGBoost — ROC Curve"
    )

    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------
    # Precision-Recall curve
    # --------------------------------------------------------

    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_test_proba,
    )

    plt.title(
        "XGBoost — Precision-Recall Curve"
    )

    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    test_metrics = calculate_metrics(
        y_test,
        y_test_pred,
        y_test_proba,
    )

    for metric, value in test_metrics.items():
        print(
            f"{metric}: "
            f"{value:.3f}"
        )


    # ========================================================
    # 13. Final summary
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("Model: XGBoost")

    print(
        f"Threshold: "
        f"{best_threshold:.3f}"
    )

    print(
        f"Best CV ROC-AUC: "
        f"{random_search.best_score_:.3f}"
    )

    print("\nTest metrics:")

    for metric, value in test_metrics.items():
        print(
            f"{metric}: "
            f"{value:.3f}"
        )


    # ========================================================
    # 14. Threshold analysis
    # ========================================================

    thresholds_to_check = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    threshold_results = []

    for threshold in thresholds_to_check:

        y_pred_threshold = (
            y_test_proba >= threshold
        ).astype(int)

        threshold_results.append({
            "Threshold": threshold,

            "Precision": precision_score(
                y_test,
                y_pred_threshold,
                zero_division=0,
            ),

            "Recall": recall_score(
                y_test,
                y_pred_threshold,
                zero_division=0,
            ),

            "F1": f1_score(
                y_test,
                y_pred_threshold,
                zero_division=0,
            ),

            "Accuracy": accuracy_score(
                y_test,
                y_pred_threshold,
            ),
        })

    threshold_df = pd.DataFrame(
        threshold_results
    )

    print("\nThreshold analysis:")

    print(
        threshold_df.to_string(
            index=False
        )
    )


    # ============================================================
    # 15. FEATURE IMPORTANCE
    # ============================================================

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    feature_importance = get_feature_importance(
        final_model
    )

    print(
        feature_importance.head(15).to_string(
            index=False
        )
    )


    # ============================================================
    # 16. PERMUTATION IMPORTANCE
    # ============================================================

    print("\n" + "=" * 60)
    print("PERMUTATION IMPORTANCE")
    print("=" * 60)

    permutation = permutation_importance(
        final_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # IMPORTANT:
    # permutation_importance() is calculated on the
    # original X_test columns because the whole pipeline
    # is passed to the function.
    permutation_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": permutation.importances_mean,
        "Std": permutation.importances_std,
    })

    permutation_df = permutation_df.sort_values(
        "Importance",
        ascending=False,
    )

    print(
        permutation_df.to_string(
            index=False
        )
    )


    # ============================================================
    # 17. SAVE FINAL MODEL
    # ============================================================

    print("\n" + "=" * 60)
    print("SAVING FINAL MODEL")
    print("=" * 60)

    os.makedirs(
        "models",
        exist_ok=True,
    )

    model_path = (
        "models/xgboost_final.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
    )

    print(
        f"Final model saved to: "
        f"{model_path}"
    )


    # ============================================================
    # 18. SAVE TEST DATA
    # ============================================================

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    test_data = X_test.copy()

    test_data["Risk"] = y_test.values

    test_data.to_csv(
        "reports/test_data.csv",
        index=False,
    )

    print(
        "Test data saved to: "
        "reports/test_data.csv"
    )


    # ============================================================
    # 19. SAVE TEST PREDICTIONS
    # ============================================================

    predictions_df = X_test.copy()

    predictions_df["true_risk"] = (
        y_test.values
    )

    predictions_df["predicted_risk"] = (
        y_test_pred
    )

    predictions_df["risk_probability"] = (
        y_test_proba
    )

    predictions_df["error_type"] = "TN"

    predictions_df.loc[
        (y_test.values == 1) &
        (y_test_pred == 1),
        "error_type"
    ] = "TP"

    predictions_df.loc[
        (y_test.values == 0) &
        (y_test_pred == 0),
        "error_type"
    ] = "TN"

    predictions_df.loc[
        (y_test.values == 0) &
        (y_test_pred == 1),
        "error_type"
    ] = "FP"

    predictions_df.loc[
        (y_test.values == 1) &
        (y_test_pred == 0),
        "error_type"
    ] = "FN"

    predictions_df.to_csv(
        "reports/test_predictions.csv",
        index=False,
    )

    print(
        "Test predictions saved to: "
        "reports/test_predictions.csv"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()