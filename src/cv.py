import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate


def create_cv(
    n_splits: int = 5,
    random_state: int = 42,
):

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def cross_validate_models(
    models,
    build_pipeline,
    X,
    y,
    cv,
):

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    results = []

    for name, model in models.items():

        print(f"\nCross-validation: {name}")

        pipeline = build_pipeline(model)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        results.append({
            "Model": name,

            "Accuracy":
                f"{scores['test_accuracy'].mean():.3f} "
                f"± {scores['test_accuracy'].std():.3f}",

            "Precision":
                f"{scores['test_precision'].mean():.3f} "
                f"± {scores['test_precision'].std():.3f}",

            "Recall":
                f"{scores['test_recall'].mean():.3f} "
                f"± {scores['test_recall'].std():.3f}",

            "F1":
                f"{scores['test_f1'].mean():.3f} "
                f"± {scores['test_f1'].std():.3f}",

            "ROC-AUC":
                f"{scores['test_roc_auc'].mean():.3f} "
                f"± {scores['test_roc_auc'].std():.3f}",
        })

    return pd.DataFrame(results)