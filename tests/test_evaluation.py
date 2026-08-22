from src.evaluation import EvaluationResult, evaluate


def test_evaluate_perfect_predictions():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]

    result = evaluate(y_true, y_pred)

    assert isinstance(result, EvaluationResult)
    assert result.accuracy == 1.0
    assert result.confusion_matrix.shape == (2, 2)


def test_evaluate_summary_contains_sections():
    result = evaluate([0, 1], [0, 1])
    summary = result.summary()

    assert "Accuracy" in summary
    assert "Confusion Matrix" in summary
    assert "Classification Report" in summary
