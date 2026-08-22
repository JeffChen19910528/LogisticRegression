from src.data import load_dataset, split_features_labels


def test_load_dataset_shape():
    df = load_dataset()
    assert list(df.columns) == ["Feature1", "Feature2", "Label"]
    assert len(df) == 10


def test_split_features_labels():
    df = load_dataset()
    X, y = split_features_labels(df)
    assert list(X.columns) == ["Feature1", "Feature2"]
    assert y.name == "Label"
    assert len(X) == len(y) == 10
