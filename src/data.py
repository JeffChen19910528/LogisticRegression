"""Dataset loading utilities."""

import pandas as pd


def load_dataset() -> pd.DataFrame:
    """Return the sample feature/label dataset as a DataFrame."""
    data = {
        "Feature1": [2.3, 1.7, 3.1, 3.5, 2.1, 1.6, 2.8, 3.0, 3.2, 2.7],
        "Feature2": [4.5, 3.2, 5.1, 5.5, 3.9, 2.4, 4.3, 4.8, 5.0, 4.1],
        "Label": [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    }
    return pd.DataFrame(data)


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into feature matrix X and label vector y."""
    X = df[["Feature1", "Feature2"]]
    y = df["Label"]
    return X, y
