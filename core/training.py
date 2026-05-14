"""Framework-free training utilities for fulfillment-rate models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from core.evaluation import evaluate_predictions
from core.features import TARGET_COLUMN, feature_columns


@dataclass(frozen=True)
class TrainTestData:
    """Train/test split container."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class ModelTrainingResult:
    """Fitted model plus evaluation outputs."""

    model: Pipeline
    metrics: dict[str, float]
    split: TrainTestData


def split_features_target(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainTestData:
    """Split a model-ready dataset into train/test features and target."""

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Dataset must contain target column: {TARGET_COLUMN}.")

    X = dataset.loc[:, feature_columns(dataset)].copy()
    y = dataset[TARGET_COLUMN].astype(float).clip(lower=0.0, upper=1.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return TrainTestData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing shared by training and inference."""

    numeric_columns = list(X.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("encoder", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_mean_baseline(X: pd.DataFrame) -> Pipeline:
    """Build a mean-prediction baseline pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )


def build_candidate_model(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """Build the first candidate tabular regression model."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "model",
                GradientBoostingRegressor(
                    loss="absolute_error",
                    n_estimators=80,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=30,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_baseline(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelTrainingResult:
    """Train and evaluate the mean baseline model."""

    split = split_features_target(dataset, test_size=test_size, random_state=random_state)
    model = build_mean_baseline(split.X_train)
    model.fit(split.X_train, split.y_train)
    metrics = evaluate_model(model, split.X_test, split.y_test)
    return ModelTrainingResult(model=model, metrics=metrics, split=split)


def train_candidate_model(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelTrainingResult:
    """Train and evaluate the first candidate model."""

    split = split_features_target(dataset, test_size=test_size, random_state=random_state)
    model = build_candidate_model(split.X_train, random_state=random_state)
    model.fit(split.X_train, split.y_train)
    metrics = evaluate_model(model, split.X_test, split.y_test)
    return ModelTrainingResult(model=model, metrics=metrics, split=split)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a fitted model on held-out data."""

    predictions = pd.Series(model.predict(X_test), index=y_test.index).clip(lower=0.0, upper=1.0)
    return evaluate_predictions(y_test, predictions)


def _one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder across scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
