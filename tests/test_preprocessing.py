from src.preprocessing import build_preprocessor


def test_ordinal_category_order_is_explicit():
    preprocessor = build_preprocessor()

    ordinal_pipeline = preprocessor.named_transformers_["ordinal"]
    ordinal_encoder = ordinal_pipeline.named_steps["ordinal"]

    assert list(ordinal_encoder.categories_[0]) == [
        "<0 DM",
        "0<=X<200 DM",
        ">=200 DM",
        "No account",
    ]

    assert list(ordinal_encoder.categories_[1]) == [
        "little",
        "moderate",
        "quite rich",
        "rich",
        "No account",
    ]
