"""Model training and prediction."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import TrainConfig


def split_train_test(X: pd.DataFrame, y: pd.Series, config: TrainConfig):
    """Split features/labels into train and test sets per config."""
    return train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Fit a logistic regression model on the training data."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
