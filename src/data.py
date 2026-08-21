import pandas as pd


def load_data(path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Risk"] = df["Risk"].map({
        "good": 0,
        "bad": 1
    })

    return df