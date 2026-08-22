"""Central configuration for the logistic regression pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.3
    random_state: int = 42
