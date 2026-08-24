import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    N_ITER_RANDOM_SEARCH,
    N_SPLITS,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    VALIDATION_SIZE,
    XGB_PARAM_GRID,
)
from .cv import create_cv, cross_validate_models
from .data import load_data, prepare_target
from .evaluation import evaluate_model, find_best_threshold, get_feature_importance
from .features import create_features
from .models import get_models
from .preprocessing import build_preprocessor


# ============================================================
# DATA SPLIT
# ============================================================


def split_data(df: pd.DataFrame):
    """Split data into train, validation and test sets."""
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

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# PIPELINE
# ============================================================


def build_pipeline(model):
    """Build preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


# ============================================================
# MAIN
# ============================================================


def main():
    df = load_data(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    df = prepare_target(df)

    print("\nTarget distribution:")
    print(df["Risk"].value_counts())

    print("\nTarget distribution (%):")
    print(df["Risk"].value_counts(normalize=True))

    df = create_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    # ========================================================
    # BASELINE MODELS
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
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

        metrics = evaluate_model(pipeline, X_val, y_val)
        metrics["Model"] = name
        baseline_results.append(metrics)

    baseline_df = pd.DataFrame(baseline_results)[
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    ]

    print("\nValidation results:")
    print(baseline_df.to_string(index=False))

    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)

    cv = create_cv(n_splits=N_SPLITS, random_state=RANDOM_STATE)

    cv_results = cross_validate_models(
        models=models,
        build_pipeline=build_pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
    )

    print("\nCross-validation results:")
    print(cv_results.to_string(index=False))

    # ========================================================
    # XGBOOST HYPERPARAMETER TUNING
    # ========================================================

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING: XGBOOST")
    print("=" * 60)

    xgb_pipeline = build_pipeline(models["XGBoost"])

    random_search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=XGB_PARAM_GRID,
        n_iter=N_ITER_RANDOM_SEARCH,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    random_search.fit(X_train, y_train)

    print("\nBest XGBoost parameters:")
    print(random_search.best_params_)
    print(f"\nBest CV ROC-AUC: {random_search.best_score_:.3f}")

    # ========================================================
    # THRESHOLD TUNING
    # ========================================================

    print("\n" + "=" * 60)
    print("THRESHOLD TUNING")
    print("=" * 60)

    tuned_xgb = random_search.best_estimator_
    y_val_proba = tuned_xgb.predict_proba(X_val)[:, 1]

    best_threshold, best_f1 = find_best_threshold(y_val, y_val_proba)

    print(f"\nBest threshold: {best_threshold:.3f}")
    print(f"Validation F1: {best_f1:.3f}")

    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    validation_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)

    print("\nTuned XGBoost validation metrics:")
    for metric, value in validation_metrics.items():
        print(f"{metric}: {value:.3f}")

    # ========================================================
    # FINAL MODEL
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL MODEL")
    print("=" * 60)

    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)

    final_model = build_pipeline(models["XGBoost"])
    final_model.set_params(**random_search.best_params_)

    print("\nTraining final XGBoost on train + validation...")
    final_model.fit(X_train_final, y_train_final)

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, cmap="Blues")
    plt.title("XGBoost — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgboost_confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_test_proba)
    plt.title("XGBoost — ROC Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgboost_roc_curve.png", dpi=200)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_test_proba)
    plt.title("XGBoost — Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgboost_precision_recall_curve.png", dpi=200)
    plt.close()

    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.3f}")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("Model: XGBoost")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Best CV ROC-AUC: {random_search.best_score_:.3f}")

    print("\nTest metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.3f}")

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    thresholds_to_check = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    threshold_results = []

    for threshold in thresholds_to_check:
        y_pred_threshold = (y_test_proba >= threshold).astype(int)
        threshold_results.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(y_test, y_pred_threshold, zero_division=0),
                "Recall": recall_score(y_test, y_pred_threshold, zero_division=0),
                "F1": f1_score(y_test, y_pred_threshold, zero_division=0),
                "Accuracy": accuracy_score(y_test, y_pred_threshold),
            }
        )

    threshold_df = pd.DataFrame(threshold_results)
    print("\nThreshold analysis:")
    print(threshold_df.to_string(index=False))
    threshold_df.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    feature_importance = get_feature_importance(final_model)
    print(feature_importance.head(15).to_string(index=False))
    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    # ========================================================
    # PERMUTATION IMPORTANCE
    # ========================================================

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

    permutation_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": permutation.importances_mean,
            "Std": permutation.importances_std,
        }
    ).sort_values("Importance", ascending=False)

    print(permutation_df.to_string(index=False))
    permutation_df.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    print("\n" + "=" * 60)
    print("SAVING FINAL MODEL")
    print("=" * 60)

    model_path = MODELS_DIR / "xgboost_final.joblib"
    joblib.dump(final_model, model_path)
    print(f"Final model saved to: {model_path}")

    # ========================================================
    # SAVE TEST DATA
    # ========================================================

    test_data = X_test.copy()
    test_data["Risk"] = y_test.values
    test_data.to_csv(REPORTS_DIR / "test_data.csv", index=False)
    print(f"Test data saved to: {REPORTS_DIR / 'test_data.csv'}")

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    predictions_df = X_test.copy()
    predictions_df["true_risk"] = y_test.values
    predictions_df["predicted_risk"] = y_test_pred
    predictions_df["risk_probability"] = y_test_proba
    predictions_df["error_type"] = "TN"

    predictions_df.loc[(y_test.values == 1) & (y_test_pred == 1), "error_type"] = "TP"
    predictions_df.loc[(y_test.values == 0) & (y_test_pred == 0), "error_type"] = "TN"
    predictions_df.loc[(y_test.values == 0) & (y_test_pred == 1), "error_type"] = "FP"
    predictions_df.loc[(y_test.values == 1) & (y_test_pred == 0), "error_type"] = "FN"

    predictions_df.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    print(f"Test predictions saved to: {REPORTS_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
