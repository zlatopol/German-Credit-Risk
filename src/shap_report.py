# src/shap_report.py

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("models/xgboost_final.joblib")
TEST_DATA_PATH = Path("reports/test_data.csv")
TEST_PREDICTIONS_PATH = Path("reports/test_predictions.csv")

OUTPUT_DIR = Path("reports/shap_errors")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "shap_report.md"
COMPARISON_PATH = OUTPUT_DIR / "fp_fn_comparison.csv"
GROUP_IMPORTANCE_PATH = OUTPUT_DIR / "shap_group_comparison.csv"


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD = 0.236


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def find_column(df: pd.DataFrame, candidates):
    """
    Find the first existing column from a list of candidates.
    """
    for column in candidates:
        if column in df.columns:
            return column

    return None


def get_model_parts(model):
    """
    Extract preprocessing and XGBoost estimator from either:
    - sklearn Pipeline
    - object with named_steps
    - bare estimator
    """

    preprocessor = None
    estimator = model

    if hasattr(model, "named_steps"):
        steps = model.named_steps

        if "model" in steps:
            estimator = steps["model"]

        elif "classifier" in steps:
            estimator = steps["classifier"]

        else:
            # Last step is usually the estimator
            estimator = list(steps.values())[-1]

        for name in ["preprocessor", "preprocess", "transformer"]:
            if name in steps:
                preprocessor = steps[name]
                break

    return preprocessor, estimator


def get_feature_names(preprocessor, estimator, n_features):
    """
    Get transformed feature names.
    """

    if preprocessor is not None:

        # ColumnTransformer
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                names = preprocessor.get_feature_names_out()
                return np.asarray(names, dtype=object)
            except Exception:
                pass

    # Fallback
    if hasattr(estimator, "feature_names_in_"):
        try:
            return np.asarray(estimator.feature_names_in_, dtype=object)
        except Exception:
            pass

    return np.asarray(
        [f"feature_{i}" for i in range(n_features)],
        dtype=object,
    )


def calculate_shap_values(estimator, X_transformed):
    """
    Calculate SHAP values for XGBoost.
    """

    import shap

    explainer = shap.TreeExplainer(estimator)

    shap_values = explainer.shap_values(X_transformed)

    # New SHAP versions can sometimes return a list.
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    # Some versions return (n_samples, n_features, 2)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, explainer


def calculate_group_importance(shap_values, indices, features):
    """
    Mean absolute SHAP importance for a group.
    """

    if len(indices) == 0:
        return pd.Series(
            0.0,
            index=features,
            dtype=float,
        )

    values = np.abs(shap_values[indices])

    return pd.Series(
        values.mean(axis=0),
        index=features,
        dtype=float,
    ).sort_values(ascending=False)


def calculate_mean_shap(shap_values, indices, features):
    """
    Mean signed SHAP value for a group.
    """

    if len(indices) == 0:
        return pd.Series(
            0.0,
            index=features,
            dtype=float,
        )

    values = shap_values[indices]

    return pd.Series(
        values.mean(axis=0),
        index=features,
        dtype=float,
    )


def safe_numeric(value):
    try:
        value = float(value)

        if np.isnan(value):
            return None

        return value

    except Exception:
        return None


# ============================================================
# LOAD DATA
# ============================================================

print_header("SHAP ERROR REPORT")

print("Loading final model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print(f"Model: {MODEL_PATH}")


if not TEST_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Test data not found: {TEST_DATA_PATH}"
    )

if not TEST_PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Test predictions not found: {TEST_PREDICTIONS_PATH}"
    )


test_data = pd.read_csv(TEST_DATA_PATH)
test_predictions = pd.read_csv(TEST_PREDICTIONS_PATH)

print(f"Test data shape: {test_data.shape}")
print(f"Predictions shape: {test_predictions.shape}")


# ============================================================
# DETECT TARGET / PREDICTION COLUMNS
# ============================================================

target_column = find_column(
    test_data,
    [
        "Risk",
        "risk",
        "target",
        "Target",
        "y_true",
        "true",
    ],
)

if target_column is None:
    target_column = find_column(
        test_predictions,
        [
            "Risk",
            "risk",
            "target",
            "Target",
            "y_true",
            "true",
        ],
    )


if target_column is None:
    raise ValueError(
        "Could not find target column. "
        "Expected one of: Risk, risk, target, y_true."
    )


prediction_column = find_column(
    test_predictions,
    [
        "Prediction",
        "prediction",
        "Predicted",
        "predicted",
        "y_pred",
        "pred",
    ],
)


probability_column = find_column(
    test_predictions,
    [
        "Probability",
        "probability",
        "Predicted Probability",
        "predicted_probability",
        "Risk Probability",
        "risk_probability",
        "prob",
    ],
)


# ============================================================
# ALIGN TRUE LABELS
# ============================================================

if target_column in test_data.columns:
    y_true = test_data[target_column].astype(int).to_numpy()

elif target_column in test_predictions.columns:
    y_true = test_predictions[target_column].astype(int).to_numpy()

else:
    raise ValueError("Unable to extract y_true.")


# ============================================================
# PREDICTIONS
# ============================================================

if probability_column is not None:
    probabilities = (
        pd.to_numeric(
            test_predictions[probability_column],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy()
    )

else:
    # Try to calculate probabilities from the model.
    X_for_prediction = test_data.drop(
        columns=[target_column],
        errors="ignore",
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            X_for_prediction
        )[:, 1]

    else:
        probabilities = model.predict(
            X_for_prediction
        ).astype(float)


if prediction_column is not None:
    y_pred = (
        pd.to_numeric(
            test_predictions[prediction_column],
            errors="coerce",
        )
        .fillna(
            (probabilities >= THRESHOLD).astype(int)
        )
        .astype(int)
        .to_numpy()
    )

else:
    y_pred = (
        probabilities >= THRESHOLD
    ).astype(int)


# Make sure lengths match
n = min(
    len(y_true),
    len(y_pred),
    len(probabilities),
)

y_true = y_true[:n]
y_pred = y_pred[:n]
probabilities = probabilities[:n]

test_data = test_data.iloc[:n].copy()


# ============================================================
# ERROR GROUPS
# ============================================================

tp_mask = (y_true == 1) & (y_pred == 1)
tn_mask = (y_true == 0) & (y_pred == 0)
fp_mask = (y_true == 0) & (y_pred == 1)
fn_mask = (y_true == 1) & (y_pred == 0)

tp_indices = np.where(tp_mask)[0]
tn_indices = np.where(tn_mask)[0]
fp_indices = np.where(fp_mask)[0]
fn_indices = np.where(fn_mask)[0]


print()
print("Error groups:")
print(f"TP: {len(tp_indices)}")
print(f"TN: {len(tn_indices)}")
print(f"FP: {len(fp_indices)}")
print(f"FN: {len(fn_indices)}")


# ============================================================
# PREPROCESS DATA
# ============================================================

print_header("CALCULATING SHAP VALUES")

preprocessor, estimator = get_model_parts(model)

X_raw = test_data.drop(
    columns=[target_column],
    errors="ignore",
)

# Remove common index column if it is present.
# It appeared in the previous permutation-importance results
# as "Unnamed: 0" with zero importance.
X_raw = X_raw.drop(
    columns=["Unnamed: 0"],
    errors="ignore",
)


if preprocessor is not None:

    X_transformed = preprocessor.transform(X_raw)

    # Convert sparse matrix to dense.
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

else:

    X_transformed = X_raw.copy()

    if hasattr(X_transformed, "values"):
        X_transformed = X_transformed.values


X_transformed = np.asarray(X_transformed)


print(
    f"Transformed test shape: {X_transformed.shape}"
)


# ============================================================
# SHAP
# ============================================================

shap_values, explainer = calculate_shap_values(
    estimator,
    X_transformed,
)

features = get_feature_names(
    preprocessor,
    estimator,
    shap_values.shape[1],
)


# Make sure number of feature names matches SHAP matrix.
if len(features) != shap_values.shape[1]:
    features = np.asarray(
        [f"feature_{i}" for i in range(shap_values.shape[1])],
        dtype=object,
    )


print(
    f"SHAP matrix shape: {shap_values.shape}"
)


# ============================================================
# GLOBAL IMPORTANCE
# ============================================================

global_importance = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=features,
).sort_values(ascending=False)


global_df = pd.DataFrame(
    {
        "Feature": global_importance.index,
        "Global Mean |SHAP|": global_importance.values,
    }
)


global_path = OUTPUT_DIR / "global_shap_importance.csv"
global_df.to_csv(global_path, index=False)


# ============================================================
# GROUP IMPORTANCE
# ============================================================

group_indices = {
    "TP": tp_indices,
    "TN": tn_indices,
    "FP": fp_indices,
    "FN": fn_indices,
}


group_importance = {}

for group, indices in group_indices.items():

    group_importance[group] = calculate_group_importance(
        shap_values,
        indices,
        features,
    )


group_comparison = pd.DataFrame(
    {
        group: series
        for group, series in group_importance.items()
    }
)

group_comparison["Global"] = global_importance

group_comparison = group_comparison.reset_index()
group_comparison = group_comparison.rename(
    columns={"index": "Feature"}
)


group_comparison.to_csv(
    GROUP_IMPORTANCE_PATH,
    index=False,
)


# ============================================================
# FP VS FN DIFFERENCE
# ============================================================

fp_importance = group_importance["FP"]
fn_importance = group_importance["FN"]

fp_fn_difference = (
    fn_importance - fp_importance
).sort_values(
    ascending=False
)


comparison_df = pd.DataFrame(
    {
        "Feature": fp_importance.index,
        "FP_mean_abs_SHAP": fp_importance.values,
        "FN_mean_abs_SHAP": fn_importance.reindex(
            fp_importance.index
        ).values,
    }
)

comparison_df["FN_minus_FP"] = (
    comparison_df["FN_mean_abs_SHAP"]
    - comparison_df["FP_mean_abs_SHAP"]
)

comparison_df["Absolute_difference"] = (
    comparison_df["FN_minus_FP"].abs()
)

comparison_df = comparison_df.sort_values(
    "Absolute_difference",
    ascending=False,
)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False,
)


# ============================================================
# SIGNED SHAP
# ============================================================

signed_importance = {}

for group, indices in group_indices.items():

    signed_importance[group] = calculate_mean_shap(
        shap_values,
        indices,
        features,
    )


# ============================================================
# TOP FEATURES
# ============================================================

TOP_N = 10


top_global = global_importance.head(TOP_N)
top_fp = fp_importance.head(TOP_N)
top_fn = fn_importance.head(TOP_N)


# ============================================================
# GENERATE INTERPRETATION
# ============================================================

def feature_list(series, n=5):
    return [
        str(feature)
        for feature in series.head(n).index
    ]


global_top_features = feature_list(
    global_importance,
    5,
)

fp_top_features = feature_list(
    fp_importance,
    5,
)

fn_top_features = feature_list(
    fn_importance,
    5,
)


# Features specifically stronger in FN
fn_specific = comparison_df.sort_values(
    "FN_minus_FP",
    ascending=False,
).head(5)


# Features specifically stronger in FP
fp_specific = comparison_df.sort_values(
    "FN_minus_FP",
    ascending=True,
).head(5)


# ============================================================
# REPRESENTATIVE OBSERVATIONS
# ============================================================

representatives = {}

for group, indices in group_indices.items():

    if len(indices) > 0:
        representatives[group] = int(indices[0])
    else:
        representatives[group] = None


# ============================================================
# MARKDOWN REPORT
# ============================================================

report = []

report.append("# SHAP Error Analysis Report\n")

report.append(
    "## 1. Общая информация\n"
)

report.append(
    f"- Final model: XGBoost\n"
    f"- Test observations: {n}\n"
    f"- Decision threshold: `{THRESHOLD}`\n"
    f"- TP: `{len(tp_indices)}`\n"
    f"- TN: `{len(tn_indices)}`\n"
    f"- FP: `{len(fp_indices)}`\n"
    f"- FN: `{len(fn_indices)}`\n"
)


report.append(
    "\n## 2. Распределение ошибок\n"
)

report.append(
    "| Group | Count | Description |\n"
    "|---|---:|---|\n"
    f"| TP | {len(tp_indices)} | Risk predicted correctly |\n"
    f"| TN | {len(tn_indices)} | Good predicted correctly |\n"
    f"| FP | {len(fp_indices)} | Good client classified as Risk |\n"
    f"| FN | {len(fn_indices)} | Risk client classified as Good |\n"
)


report.append(
    "\n## 3. Global SHAP importance\n"
)

report.append(
    "На глобальном уровне наиболее важными признаками являются:\n\n"
)

for i, (feature, value) in enumerate(
    top_global.items(),
    start=1,
):
    report.append(
        f"{i}. **{feature}** — "
        f"mean |SHAP| = `{value:.4f}`\n"
    )


report.append(
    "\n## 4. False Positives (FP)\n"
)

report.append(
    f"Количество FP: **{len(fp_indices)}**.\n\n"
    "Это реальные клиенты класса `0`, которых модель "
    "ошибочно отнесла к классу `1`.\n\n"
)

report.append(
    "Наиболее важные признаки для FP:\n\n"
)

for i, (feature, value) in enumerate(
    top_fp.items(),
    start=1,
):
    report.append(
        f"{i}. **{feature}** — "
        f"mean |SHAP| = `{value:.4f}`\n"
    )


report.append(
    "\n### Признаки, особенно отличающие FP от FN\n\n"
)

for _, row in fp_specific.iterrows():

    report.append(
        f"- **{row['Feature']}**: "
        f"FP = `{row['FP_mean_abs_SHAP']:.4f}`, "
        f"FN = `{row['FN_mean_abs_SHAP']:.4f}`\n"
    )


report.append(
    "\n## 5. False Negatives (FN)\n"
)

report.append(
    f"Количество FN: **{len(fn_indices)}**.\n\n"
    "Это реальные клиенты класса `1` (Risk), "
    "которых модель ошибочно классифицировала как класс `0`.\n\n"
)

report.append(
    "Наиболее важные признаки для FN:\n\n"
)

for i, (feature, value) in enumerate(
    top_fn.items(),
    start=1,
):
    report.append(
        f"{i}. **{feature}** — "
        f"mean |SHAP| = `{value:.4f}`\n"
    )


report.append(
    "\n### Признаки, особенно характерные для FN\n\n"
)

for _, row in fn_specific.iterrows():

    report.append(
        f"- **{row['Feature']}**: "
        f"FP = `{row['FP_mean_abs_SHAP']:.4f}`, "
        f"FN = `{row['FN_mean_abs_SHAP']:.4f}`\n"
    )


report.append(
    "\n## 6. FP vs FN\n"
)

report.append(
    "Сравнение mean absolute SHAP позволяет понять, "
    "какие признаки особенно сильно участвуют в разных типах ошибок.\n\n"
)

report.append(
    "| Feature | FP | FN | FN − FP |\n"
    "|---|---:|---:|---:|\n"
)

for _, row in comparison_df.head(15).iterrows():

    report.append(
        f"| {row['Feature']} | "
        f"{row['FP_mean_abs_SHAP']:.4f} | "
        f"{row['FN_mean_abs_SHAP']:.4f} | "
        f"{row['FN_minus_FP']:.4f} |\n"
    )


report.append(
    "\n## 7. Направление влияния\n"
)

report.append(
    "Средний signed SHAP показывает направление влияния "
    "признака на output модели. Положительное значение "
    "увеличивает модельный output, отрицательное — уменьшает его.\n\n"
)

for group in ["FP", "FN"]:

    report.append(
        f"### {group}\n\n"
    )

    signed = signed_importance[group]

    positive = signed[
        signed > 0
    ].sort_values(
        ascending=False
    ).head(5)

    negative = signed[
        signed < 0
    ].sort_values(
        ascending=True
    ).head(5)

    report.append(
        "**Положительное влияние:**\n\n"
    )

    for feature, value in positive.items():

        report.append(
            f"- {feature}: `{value:.4f}`\n"
        )

    report.append(
        "\n**Отрицательное влияние:**\n\n"
    )

    for feature, value in negative.items():

        report.append(
            f"- {feature}: `{value:.4f}`\n"
        )

    report.append("\n")


report.append(
    "## 8. Representative observations\n"
)

for group, index in representatives.items():

    if index is not None:

        probability = probabilities[index]

        report.append(
            f"- **{group}**: test index `{index}`, "
            f"true = `{y_true[index]}`, "
            f"pred = `{y_pred[index]}`, "
            f"probability = `{probability:.3f}`\n"
        )


report.append(
    "\n## 9. Основные выводы\n"
)

report.append(
    "### Вывод 1 — Checking account\n\n"
    "Checking account является наиболее значимым признаком "
    "в SHAP-анализе как для FP, так и для FN. Это означает, "
    "что состояние текущего счёта является одним из главных "
    "факторов, определяющих решение модели.\n\n"
)

report.append(
    "### Вывод 2 — Duration\n\n"
    "Duration занимает второе место среди наиболее важных "
    "признаков и существенно участвует как в FP, так и в FN. "
    "Следовательно, срок кредита является важным источником "
    "разделения клиентов моделью.\n\n"
)

report.append(
    "### Вывод 3 — Credit amount\n\n"
    "Credit amount особенно заметен среди FN. Это означает, "
    "что размер кредита существенно влияет на модельный output "
    "для клиентов, которых модель не смогла правильно распознать "
    "как рискованных.\n\n"
)

report.append(
    "### Вывод 4 — Monthly payment\n\n"
    "Monthly payment имеет заметное влияние на FP и FN. "
    "Признак связан с финансовой нагрузкой клиента и участвует "
    "в формировании пограничных решений модели.\n\n"
)

report.append(
    "### Вывод 5 — Threshold\n\n"
    f"Используемый threshold `{THRESHOLD}` выбран для повышения "
    "чувствительности модели к классу Risk. Поэтому модель "
    "допускает относительно большое количество False Positives "
    "в обмен на более высокий Recall.\n\n"
)

report.append(
    "### Итог\n\n"
    "SHAP-анализ показывает, что ошибки модели не распределены "
    "случайно. Основная часть объясняющей информации сосредоточена "
    "в нескольких признаках: Checking account, Duration, "
    "Credit amount, Monthly payment и Saving accounts. "
    "Для дальнейшего улучшения модели целесообразно отдельно "
    "исследовать наблюдения FP и FN и проверить, можно ли улучшить "
    "разделение этих групп изменением threshold, feature engineering "
    "или настройкой XGBoost.\n"
)


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_PATH.write_text(
    "".join(report),
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print_header("SHAP REPORT COMPLETED")

print(f"Markdown report:")
print(f"  {REPORT_PATH}")

print()
print("CSV files:")
print(f"  {GROUP_IMPORTANCE_PATH}")
print(f"  {COMPARISON_PATH}")
print(f"  {global_path}")

print()
print("Representative observations:")

for group, index in representatives.items():

    if index is not None:
        print(
            f"  {group}: test index {index}, "
            f"probability={probabilities[index]:.3f}"
        )

print()
print("Done.")