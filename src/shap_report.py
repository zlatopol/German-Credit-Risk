"""Generate the Markdown SHAP error-analysis report.

Run with::

    python -m src.shap_report

SHAP calculation and model/preprocessing handling live in ``shap_analysis``;
this module is responsible only for aggregation, interpretation and report
writing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DECISION_THRESHOLD, REPORTS_DIR
from .shap_analysis import (
    calculate_group_importance,
    calculate_mean_shap,
    calculate_shap_values,
    create_error_groups,
    get_predictions,
    load_model_and_test_data,
)


OUTPUT_DIR = REPORTS_DIR / "shap_errors"
REPORT_PATH = OUTPUT_DIR / "shap_report.md"
GLOBAL_IMPORTANCE_PATH = OUTPUT_DIR / "global_shap_importance.csv"
GROUP_IMPORTANCE_PATH = OUTPUT_DIR / "shap_group_comparison.csv"
COMPARISON_PATH = OUTPUT_DIR / "fp_fn_comparison.csv"

TOP_N = 10


def print_header(title: str):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def build_group_importance_table(shap_values, feature_names, error_groups):
    """Build TP/TN/FP/FN and global mean-|SHAP| comparison table."""

    global_importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
        dtype=float,
    )

    group_series = {
        group: calculate_group_importance(
            shap_values,
            feature_names,
            mask,
        ).set_index("Feature")["Importance"]
        for group, mask in error_groups.items()
    }

    table = pd.DataFrame(group_series)
    table["Global"] = global_importance
    table.index.name = "Feature"

    return table.reset_index(), global_importance, group_series


def build_fp_fn_comparison(group_series):
    """Compare mean absolute SHAP importance between FP and FN."""

    features = group_series["FP"].index
    comparison = pd.DataFrame(
        {
            "Feature": features,
            "FP_mean_abs_SHAP": group_series["FP"].reindex(features).values,
            "FN_mean_abs_SHAP": group_series["FN"].reindex(features).values,
        }
    )

    comparison["FN_minus_FP"] = (
        comparison["FN_mean_abs_SHAP"]
        - comparison["FP_mean_abs_SHAP"]
    )
    comparison["Absolute_difference"] = comparison["FN_minus_FP"].abs()

    return comparison.sort_values(
        "Absolute_difference",
        ascending=False,
    ).reset_index(drop=True)


def build_signed_shap(shap_values, feature_names, error_groups):
    """Calculate mean signed SHAP values for every error group."""

    return {
        group: calculate_mean_shap(
            shap_values,
            feature_names,
            mask,
        )
        for group, mask in error_groups.items()
    }


def build_report(
    y_true,
    y_pred,
    probabilities,
    error_groups,
    global_importance,
    group_series,
    signed_importance,
    representatives,
    threshold,
):
    """Build the complete Markdown report."""

    comparison = build_fp_fn_comparison(group_series)
    top_global = global_importance.head(TOP_N)
    top_fp = group_series["FP"].head(TOP_N)
    top_fn = group_series["FN"].head(TOP_N)
    fn_specific = comparison.sort_values("FN_minus_FP", ascending=False).head(5)
    fp_specific = comparison.sort_values("FN_minus_FP", ascending=True).head(5)

    tp_count = int(error_groups["TP"].sum())
    tn_count = int(error_groups["TN"].sum())
    fp_count = int(error_groups["FP"].sum())
    fn_count = int(error_groups["FN"].sum())

    report = [
        "# SHAP Error Analysis Report\n",
        "## 1. Общая информация\n",
        "- Final model: XGBoost\n",
        f"- Test observations: `{len(y_true)}`\n",
        f"- Decision threshold: `{threshold}`\n",
        f"- TP: `{tp_count}`\n",
        f"- TN: `{tn_count}`\n",
        f"- FP: `{fp_count}`\n",
        f"- FN: `{fn_count}`\n",
        "\n## 2. Распределение ошибок\n",
        "| Group | Count | Description |\n",
        "|---|---:|---|\n",
        f"| TP | {tp_count} | Risk predicted correctly |\n",
        f"| TN | {tn_count} | Good predicted correctly |\n",
        f"| FP | {fp_count} | Good client classified as Risk |\n",
        f"| FN | {fn_count} | Risk client classified as Good |\n",
        "\n## 3. Global SHAP importance\n",
        "На глобальном уровне наиболее важными признаками являются:\n\n",
    ]

    for i, (feature, value) in enumerate(top_global.items(), start=1):
        report.append(f"{i}. **{feature}** — mean |SHAP| = `{value:.4f}`\n")

    report.extend([
        "\n## 4. False Positives (FP)\n",
        f"Количество FP: **{fp_count}**.\n\n",
        "Это реальные клиенты класса `0`, которых модель ошибочно отнесла к классу `1`.\n\n",
        "Наиболее важные признаки для FP:\n\n",
    ])

    for i, (feature, value) in enumerate(top_fp.items(), start=1):
        report.append(f"{i}. **{feature}** — mean |SHAP| = `{value:.4f}`\n")

    report.append("\n### Признаки, особенно отличающие FP от FN\n\n")
    for _, row in fp_specific.iterrows():
        report.append(
            f"- **{row['Feature']}**: FP = `{row['FP_mean_abs_SHAP']:.4f}`, "
            f"FN = `{row['FN_mean_abs_SHAP']:.4f}`\n"
        )

    report.extend([
        "\n## 5. False Negatives (FN)\n",
        f"Количество FN: **{fn_count}**.\n\n",
        "Это реальные клиенты класса `1` (Risk), которых модель ошибочно классифицировала как класс `0`.\n\n",
        "Наиболее важные признаки для FN:\n\n",
    ])

    for i, (feature, value) in enumerate(top_fn.items(), start=1):
        report.append(f"{i}. **{feature}** — mean |SHAP| = `{value:.4f}`\n")

    report.append("\n### Признаки, особенно характерные для FN\n\n")
    for _, row in fn_specific.iterrows():
        report.append(
            f"- **{row['Feature']}**: FP = `{row['FP_mean_abs_SHAP']:.4f}`, "
            f"FN = `{row['FN_mean_abs_SHAP']:.4f}`\n"
        )

    report.extend([
        "\n## 6. FP vs FN\n",
        "Сравнение mean absolute SHAP позволяет понять, какие признаки особенно сильно участвуют в разных типах ошибок.\n\n",
        "| Feature | FP | FN | FN − FP |\n",
        "|---|---:|---:|---:|\n",
    ])

    for _, row in comparison.head(15).iterrows():
        report.append(
            f"| {row['Feature']} | {row['FP_mean_abs_SHAP']:.4f} | "
            f"{row['FN_mean_abs_SHAP']:.4f} | {row['FN_minus_FP']:.4f} |\n"
        )

    report.extend([
        "\n## 7. Направление влияния\n",
        "Средний signed SHAP показывает направление влияния признака на output модели. "
        "Положительное значение увеличивает model output, отрицательное — уменьшает его.\n\n",
    ])

    for group in ("FP", "FN"):
        signed = signed_importance[group]
        positive = signed[signed > 0].sort_values(ascending=False).head(5)
        negative = signed[signed < 0].sort_values().head(5)

        report.append(f"### {group}\n\n")
        report.append("**Положительное влияние:**\n\n")
        for feature, value in positive.items():
            report.append(f"- {feature}: `{value:.4f}`\n")

        report.append("\n**Отрицательное влияние:**\n\n")
        for feature, value in negative.items():
            report.append(f"- {feature}: `{value:.4f}`\n")
        report.append("\n")

    report.append("## 8. Representative observations\n")
    for group, index in representatives.items():
        if index is not None:
            report.append(
                f"- **{group}**: test index `{index}`, true = `{y_true[index]}`, "
                f"pred = `{y_pred[index]}`, probability = `{probabilities[index]:.3f}`\n"
            )

    report.extend([
        "\n## 9. Основные выводы\n",
        "### Вывод 1 — Checking account\n\n",
        "Checking account является наиболее значимым признаком в SHAP-анализе как для FP, так и для FN. Это означает, что состояние текущего счёта является одним из главных факторов, определяющих решение модели.\n\n",
        "### Вывод 2 — Duration\n\n",
        "Duration занимает второе место среди наиболее важных признаков и существенно участвует как в FP, так и в FN. Следовательно, срок кредита является важным источником разделения клиентов моделью.\n\n",
        "### Вывод 3 — Credit amount\n\n",
        "Credit amount особенно заметен среди FN. Это означает, что размер кредита существенно влияет на модельный output для клиентов, которых модель не смогла правильно распознать как рискованных.\n\n",
        "### Вывод 4 — Monthly payment\n\n",
        "Monthly payment имеет заметное влияние на FP и FN. Признак связан с финансовой нагрузкой клиента и участвует в формировании пограничных решений модели.\n\n",
        "### Вывод 5 — Threshold\n\n",
        f"Используемый threshold `{threshold}` выбран для повышения чувствительности модели к классу Risk. Поэтому модель допускает относительно большое количество False Positives в обмен на более высокий Recall.\n\n",
        "### Итог\n\n",
        "SHAP-анализ показывает, что ошибки модели не распределены случайно. Основная часть объясняющей информации сосредоточена в нескольких признаках: Checking account, Duration, Credit amount, Monthly payment и Saving accounts. Для дальнейшего улучшения модели целесообразно отдельно исследовать наблюдения FP и FN и проверить, можно ли улучшить разделение этих групп изменением threshold, feature engineering или настройкой XGBoost.\n",
    ])

    return "".join(report)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_header("SHAP ERROR REPORT")
    print("Loading final model and test data...")

    model_pipeline, X_test, y_test = load_model_and_test_data()
    print(f"Test shape: {X_test.shape}")

    print("\nGenerating predictions...")
    y_proba, y_pred = get_predictions(
        model_pipeline,
        X_test,
        threshold=DECISION_THRESHOLD,
    )

    error_groups = create_error_groups(y_test, y_pred)

    print("\nError groups:")
    for group, mask in error_groups.items():
        print(f"{group}: {mask.sum()}")

    print_header("CALCULATING SHAP VALUES")
    _, shap_values, _, _, feature_names = calculate_shap_values(
        model_pipeline,
        X_test,
    )

    _, global_importance, group_series = build_group_importance_table(
        shap_values,
        feature_names,
        error_groups,
    )

    comparison = build_fp_fn_comparison(group_series)
    signed_importance = build_signed_shap(
        shap_values,
        feature_names,
        error_groups,
    )

    representatives = {
        group: (int(np.where(mask)[0][0]) if mask.sum() else None)
        for group, mask in error_groups.items()
    }

    pd.DataFrame(
        {
            "Feature": global_importance.index,
            "Global Mean |SHAP|": global_importance.values,
        }
    ).to_csv(GLOBAL_IMPORTANCE_PATH, index=False)

    group_table, _, _ = build_group_importance_table(
        shap_values,
        feature_names,
        error_groups,
    )
    group_table.to_csv(GROUP_IMPORTANCE_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)

    REPORT_PATH.write_text(
        build_report(
            np.asarray(y_test),
            y_pred,
            y_proba,
            error_groups,
            global_importance,
            group_series,
            signed_importance,
            representatives,
            DECISION_THRESHOLD,
        ),
        encoding="utf-8",
    )

    print_header("SHAP REPORT COMPLETED")
    print(f"Markdown report: {REPORT_PATH}")
    print("\nCSV files:")
    print(f"  {GLOBAL_IMPORTANCE_PATH}")
    print(f"  {GROUP_IMPORTANCE_PATH}")
    print(f"  {COMPARISON_PATH}")
    print(f"\nThreshold: {DECISION_THRESHOLD}")


if __name__ == "__main__":
    main()
