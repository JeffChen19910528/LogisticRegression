"""Entry point wiring data loading, training, and evaluation together."""

from src.config import TrainConfig
from src.data import load_dataset, split_features_labels
from src.evaluation import evaluate
from src.model import split_train_test, train_model


def run(config: TrainConfig = TrainConfig()) -> None:
    df = load_dataset()
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, config)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    result = evaluate(y_test, y_pred)
    print(result.summary())


if __name__ == "__main__":
    run()
