# Logistic Regression Introduction

A small, modular example project demonstrating logistic regression with scikit-learn.

## Project Structure

```
.
├── src/
│   ├── config.py      # Training configuration (test size, random state)
│   ├── data.py         # Dataset loading and feature/label splitting
│   ├── model.py         # Train/test splitting and model training
│   ├── evaluation.py    # Accuracy, confusion matrix, classification report
│   └── main.py           # Wires everything together and runs the pipeline
├── tests/                 # Unit tests for each module
├── requirements.txt       # Runtime dependencies
└── requirements-dev.txt   # Runtime + test dependencies
```

Each module has a single responsibility and depends only on the modules it needs, making the pipeline easy to test and extend independently (e.g. swapping the dataset or the model).

## Installation

```bash
pip install -r requirements.txt
```

For running tests, install dev dependencies instead:

```bash
pip install -r requirements-dev.txt
```

## Usage

Run the full pipeline (load data, train, evaluate):

```bash
python -m src.main
```

## Testing

```bash
pytest
```
