import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Estimated monthly loan payment
    df["Monthly payment"] = (
        df["Credit amount"] / df["Duration"]
    )

    # Log-transformed monthly payment
    df["Log Monthly payment"] = np.log1p(
        df["Monthly payment"]
    )

    # Estimated age at the end of the loan
    df["Age after loan"] = (
        df["Age"] + df["Duration"] / 12
    )

    # Monthly payment relative to age
    df["Payment_to_age"] = (
        df["Monthly payment"] / df["Age"]
    )

    # Young customer indicator
    df["Young"] = (
        df["Age"] < 25
    ).astype(int)

    return df