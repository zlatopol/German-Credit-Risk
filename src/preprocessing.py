from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def build_preprocessor() -> ColumnTransformer:

    numeric_features = [
        "Age",
        "Duration",
        "Credit amount",
        "Monthly payment",
        "Log Monthly payment",
        "Age after loan",
        "Payment_to_age",
    ]

    categorical_features = [
        "Sex",
        "Housing",
        "Purpose",
    ]

    ordinal_features = [
        "Saving accounts",
        "Checking account",
    ]

    binary_features = [
        "Young",
    ]

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "ohe",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ])

    ordinal_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="No account",
            ),
        ),
        (
            "ordinal",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    numeric_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "ordinal",
                ordinal_pipeline,
                ordinal_features,
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "binary",
                "passthrough",
                binary_features,
            ),
        ]
    )

    return preprocessor