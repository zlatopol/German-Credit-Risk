import numpy as np
import pandas as pd

from src.features import create_features


def test_create_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "Age": [20.0, 40.0],
            "Duration": [12.0, 24.0],
            "Credit amount": [1200.0, 2400.0],
        }
    )

    result = create_features(df)

    expected = {
        "Monthly payment",
        "Log Monthly payment",
        "Age after loan",
        "Payment_to_age",
        "Young",
    }

    assert expected.issubset(result.columns)


def test_create_features_calculates_values_correctly():
    df = pd.DataFrame(
        {
            "Age": [20.0, 40.0],
            "Duration": [12.0, 24.0],
            "Credit amount": [1200.0, 2400.0],
        }
    )

    result = create_features(df)

    np.testing.assert_allclose(
        result["Monthly payment"],
        [100.0, 100.0],
    )
    np.testing.assert_allclose(
        result["Log Monthly payment"],
        np.log1p([100.0, 100.0]),
    )
    np.testing.assert_allclose(
        result["Age after loan"],
        [21.0, 42.0],
    )
    np.testing.assert_allclose(
        result["Payment_to_age"],
        [5.0, 2.5],
    )
    assert result["Young"].tolist() == [1, 0]
