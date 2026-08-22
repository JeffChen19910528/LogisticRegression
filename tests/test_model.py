from sklearn.linear_model import LogisticRegression

from src.config import TrainConfig
from src.data import load_dataset, split_features_labels
from src.model import split_train_test, train_model


def test_split_train_test_respects_config():
    df = load_dataset()
    X, y = split_features_labels(df)
    config = TrainConfig(test_size=0.3, random_state=42)

    X_train, X_test, y_train, y_test = split_train_test(X, y, config)

    assert len(X_test) == 3
    assert len(X_train) == 7
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_train_model_returns_fitted_estimator():
    df = load_dataset()
    X, y = split_features_labels(df)
    config = TrainConfig()
    X_train, _, y_train, _ = split_train_test(X, y, config)

    model = train_model(X_train, y_train)

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
