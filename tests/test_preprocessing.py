from src.preprocessing import build_preprocessor


def test_ordinal_category_order_is_explicit():
    preprocessor = build_preprocessor()

    ordinal_pipeline = next(
        transformer
        for name, transformer, columns in preprocessor.transformers
        if name == "ordinal"
    )

    ordinal_encoder = ordinal_pipeline.named_steps["ordinal"]

    assert ordinal_encoder.categories[0] == [
        "<0 DM",
        "0<=X<200 DM",
        ">=200 DM",
        "No account",
    ]

    assert ordinal_encoder.categories[1] == [
        "little",
        "moderate",
        "quite rich",
        "rich",
        "No account",
    ]