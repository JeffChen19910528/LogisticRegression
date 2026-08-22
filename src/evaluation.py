"""Model evaluation utilities."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class EvaluationResult:
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: str

    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.2f}\n"
            "Confusion Matrix:\n"
            f"{self.confusion_matrix}\n"
            "Classification Report:\n"
            f"{self.classification_report}"
        )


def evaluate(y_true, y_pred) -> EvaluationResult:
    """Compute accuracy, confusion matrix, and classification report."""
    return EvaluationResult(
        accuracy=accuracy_score(y_true, y_pred),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        classification_report=classification_report(y_true, y_pred, zero_division=0),
    )
