"""Reusable preprocessing entry point for the mobile usage ML project.

This module is the single place where "raw processed dataset -> model-ready
train/test split + preprocessing pipeline" logic lives. It reproduces, without
duplication, the exact preprocessing decisions validated in the research
notebooks:

- ``02_feature_engineering.ipynb`` -- which columns are redundant / reserved
  for later tasks.
- ``03_regression.ipynb`` sections 4-7 -- leakage exclusions, train/test
  split configuration, automatic feature-type detection, and the
  numerical/categorical preprocessing pipelines.

Intended consumers (all still to be built): classification training,
clustering, the FastAPI service, the Streamlit app, and batch inference jobs.
Every consumer should call into this module rather than re-implementing
imputation, scaling, or encoding logic locally.

Explicitly out of scope for this module:

- No model training or hyperparameter search.
- No exploratory data analysis or plotting.
- No notebook-specific variables or one-off analysis code.
- No API, UI, or deployment code.

Public API
----------
- ``load_processed_data``       -- read the processed dataset from disk.
- ``get_exclude_columns``       -- resolve ID/redundant/leakage columns to drop.
- ``detect_feature_types``      -- dtype-based numerical/categorical detection.
- ``split_features_target``     -- separate ``X`` / ``y`` for a given target.
- ``split_train_test``          -- train/test split wrapper.
- ``build_numerical_pipeline``  -- median impute + scale.
- ``build_categorical_pipeline``-- impute + one-hot encode.
- ``build_preprocessing_pipeline`` -- combine both into a ``ColumnTransformer``.
- ``prepare_dataset``           -- end-to-end convenience orchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class PreprocessingError(Exception):
    """Base exception for all preprocessing-related errors in this module."""

class DatasetNotFoundError(PreprocessingError, FileNotFoundError):
    """Raised when the requested processed dataset file cannot be found."""

class EmptyDatasetError(PreprocessingError, ValueError):
    """Raised when a loaded dataset contains zero rows."""

class TargetColumnNotFoundError(PreprocessingError, ValueError):
    """Raised when the requested target column is not present in the dataset."""

class UnsupportedFileFormatError(PreprocessingError, ValueError):
    """Raised when the dataset file extension is not supported by the loader."""

# --------------------------------------------------------------------------
# Constants -- mirror the exact values validated in the notebooks
# --------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Resolved relative to this file's location (``src/data/preprocess.py``),
    so it does not depend on the caller's current working directory and
    never hardcodes an absolute, machine-specific path.
    """
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = get_project_root()
DEFAULT_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# Produced by 02_feature_engineering.ipynb and validated in 03_regression.ipynb.
DEFAULT_DATA_FILENAME = "mobile_app_usage_features.csv"

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MIN_FREQUENCY = 0.01

# Identifier columns: never predictive, always excluded from any feature matrix.
DEFAULT_ID_COLUMNS: List[str] = ["record_id"]

# Columns that are a deterministic re-encoding of another column already kept,
# regardless of which target is being predicted. `age_group` duplicates
# `age_group_numeric` (see 02_feature_engineering.ipynb section 5.3 / 4).
DEFAULT_REDUNDANT_COLUMNS: List[str] = ["age_group"]

# Target-specific exclusions: columns that would leak the target or duplicate
# it under another name. Keyed by target column name.
#
# "screen_time_hours" reproduces exactly the `DROP_FROM_X` list validated in
# 03_regression.ipynb section 4 (minus the target itself, added separately):
#   - "daily_screen_time_minutes" is the direct numeric source of the target.
#   - "app_deleted_and_reinstalled" / "target" were reserved for the
#     (not yet built) classification notebook to keep notebook scope clean.
#
# "target" / "app_deleted_and_reinstalled" are the two encodings of the same
# classification label (see 02_feature_engineering.ipynb section 4) -- each
# must exclude the other to avoid trivially leaking itself.
#
# Targets not listed here have not been validated by a notebook yet; see
# `get_exclude_columns` for the fallback behaviour.
TARGET_SPECIFIC_EXCLUDE_COLUMNS: Dict[str, List[str]] = {
    "screen_time_hours": ["daily_screen_time_minutes", "app_deleted_and_reinstalled", "target"],
    "target": ["app_deleted_and_reinstalled"],
    "app_deleted_and_reinstalled": ["target"],
}

# Targets that are a pure unit/format conversion of a stored column rather
# than a column physically present in the processed file. Reproduces the
# derivation validated in 03_regression.ipynb section 4
# (`screen_time_hours = daily_screen_time_minutes / 60`) -- this is a unit
# conversion of the label itself, not new feature engineering.
TARGET_DERIVATION_FUNCS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "screen_time_hours": lambda df: df["daily_screen_time_minutes"] / 60,
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_processed_data(
    filename: str = DEFAULT_DATA_FILENAME,
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load a processed dataset from the ``data/processed`` directory.

    Args:
        filename: Name of the file inside the processed data directory.
            Defaults to the dataset produced by ``02_feature_engineering.ipynb``
            and used throughout ``03_regression.ipynb``.
        processed_dir: Optional override for the processed data directory.
            Defaults to ``<project_root>/data/processed``, resolved via
            :func:`get_project_root` (no absolute path is hardcoded).

    Returns:
        The loaded dataset as a pandas DataFrame.

    Raises:
        DatasetNotFoundError: If the resolved file path does not exist.
        EmptyDatasetError: If the file exists but contains zero rows.
        UnsupportedFileFormatError: If the file extension is neither
            ``.csv`` nor ``.parquet``.
    """
    directory = processed_dir if processed_dir is not None else DEFAULT_PROCESSED_DIR
    file_path = directory / filename

    if not file_path.exists():
        raise DatasetNotFoundError(
            f"Processed dataset not found at '{file_path}'. "
            "Run 02_feature_engineering.ipynb first to generate it."
        )

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise UnsupportedFileFormatError(
            f"Unsupported file format '{suffix}' for '{file_path}'. "
            "Expected '.csv' or '.parquet'."
        )

    if df.empty:
        raise EmptyDatasetError(f"Dataset loaded from '{file_path}' contains zero rows.")

    logger.info("Loaded processed dataset from %s (%d rows, %d columns).", file_path, *df.shape)
    return df


# --------------------------------------------------------------------------
# Target derivation / column exclusion
# --------------------------------------------------------------------------


def _ensure_target_column(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Derive ``target`` if it is missing from ``df`` but derivable.

    Some targets used across the notebooks (e.g. ``screen_time_hours``) are a
    pure unit conversion of a column stored in the processed file rather than
    a stored column themselves. If a derivation is registered in
    :data:`TARGET_DERIVATION_FUNCS`, it is applied on a copy of ``df``;
    otherwise ``df`` is returned unchanged and any missing-column error is
    left to the caller (:func:`split_features_target`) to raise.
    """
    if target in df.columns:
        return df

    deriver = TARGET_DERIVATION_FUNCS.get(target)
    if deriver is None:
        return df

    df = df.copy()
    df[target] = deriver(df)
    logger.info("Derived target column '%s' (not physically present in source file).", target)
    return df


def get_exclude_columns(target: str, extra_exclude: Optional[Sequence[str]] = None) -> List[str]:
    """Resolve the full list of columns to exclude from a feature matrix.

    Combines, in order:

    1. The target column itself.
    2. :data:`DEFAULT_ID_COLUMNS` -- identifiers carry no predictive signal.
    3. :data:`DEFAULT_REDUNDANT_COLUMNS` -- deterministic duplicates of other
       kept columns, regardless of target.
    4. :data:`TARGET_SPECIFIC_EXCLUDE_COLUMNS` for this ``target`` --
       leakage / reserved-for-another-task columns. If ``target`` has no
       registered entry, a warning is logged: the combination has not been
       validated by a notebook and only ID + redundant columns are excluded
       by default.
    5. ``extra_exclude`` -- caller-supplied additions.

    Args:
        target: Name of the target column the caller intends to predict.
        extra_exclude: Additional columns to exclude, on top of the above.

    Returns:
        A de-duplicated list of column names to drop from the feature matrix.
    """
    if target not in TARGET_SPECIFIC_EXCLUDE_COLUMNS:
        logger.warning(
            "No validated exclusion list registered for target '%s'. Falling back to "
            "ID + redundant columns only -- review for leakage manually before using "
            "this target in production.",
            target,
        )

    combined = (
        [target]
        + DEFAULT_ID_COLUMNS
        + DEFAULT_REDUNDANT_COLUMNS
        + TARGET_SPECIFIC_EXCLUDE_COLUMNS.get(target, [])
        + list(extra_exclude or [])
    )

    seen: set = set()
    deduped: List[str] = []
    for col in combined:
        if col not in seen:
            seen.add(col)
            deduped.append(col)
    return deduped


# --------------------------------------------------------------------------
# Feature detection / feature-target split
# --------------------------------------------------------------------------


def detect_feature_types(
    df: pd.DataFrame, exclude: Optional[Sequence[str]] = None
) -> Tuple[List[str], List[str]]:
    """Automatically detect numerical and categorical feature columns.

    Detection is purely dtype-based (no column names are hardcoded), exactly
    as validated in ``03_regression.ipynb`` section 6.

    Args:
        df: The DataFrame to inspect (typically a feature matrix ``X``).
        exclude: Columns to leave out of both returned lists.

    Returns:
        A tuple ``(numerical_features, categorical_features)``.

    Raises:
        PreprocessingError: If any remaining column has a dtype that is
            neither numeric nor object/category (e.g. datetime, bool, since
            such columns need an explicit handling decision that this
            function cannot make safely on its own).
    """
    exclude_set = set(exclude or [])
    candidate_columns = [col for col in df.columns if col not in exclude_set]
    subset = df[candidate_columns]

    # "string" covers pandas' newer native StringDtype (default for text columns
    # read from CSV as of pandas >= 3.0); "object" keeps supporting the legacy
    # representation used by older pandas versions. Both are included
    # explicitly to avoid a pandas 3.x FutureWarning about implicit coverage.
    numerical_features = subset.select_dtypes(include=np.number).columns.tolist()
    categorical_features = subset.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    covered = set(numerical_features) | set(categorical_features)
    uncovered = sorted(set(candidate_columns) - covered)
    if uncovered:
        dtypes = [str(df[col].dtype) for col in uncovered]
        raise PreprocessingError(
            f"Columns with unsupported dtypes require an explicit decision before use "
            f"as features: {uncovered} (dtypes: {dtypes})."
        )

    logger.info(
        "Detected %d numerical and %d categorical features.",
        len(numerical_features),
        len(categorical_features),
    )
    return numerical_features, categorical_features


def split_features_target(
    df: pd.DataFrame,
    target: str,
    extra_exclude: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into a feature matrix ``X`` and target vector ``y``.

    Args:
        df: The full dataset. If ``target`` is a derivable target (see
            :data:`TARGET_DERIVATION_FUNCS`) it does not need to already be
            a column of ``df``.
        target: Name of the target column.
        extra_exclude: Additional non-target columns to drop from ``X``
            beyond the defaults resolved by :func:`get_exclude_columns`.

    Returns:
        A tuple ``(X, y)``.

    Raises:
        TargetColumnNotFoundError: If ``target`` is neither a column of
            ``df`` nor derivable from it.
    """
    df = _ensure_target_column(df, target)

    if target not in df.columns:
        raise TargetColumnNotFoundError(
            f"Target column '{target}' not found in dataset and no derivation is "
            f"registered for it. Available columns: {sorted(df.columns)}"
        )

    exclude_columns = get_exclude_columns(target, extra_exclude=extra_exclude)
    y = df[target]
    X = df.drop(columns=[col for col in exclude_columns if col in df.columns])

    logger.info(
        "Split dataset into X (%d rows, %d columns) and y ('%s').",
        X.shape[0],
        X.shape[1],
        target,
    )
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features/target into train and test subsets.

    Mirrors the split validated in ``03_regression.ipynb`` section 5 (80/20,
    ``random_state=42``). A train/test split is required because all
    preprocessing statistics (imputation values, scaler mean/std, one-hot
    categories) must be learned only from the training data -- evaluating
    on data the pipeline has never seen is what makes reported metrics a
    realistic estimate of production performance rather than a measure of
    memorization.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of rows reserved for the test set.
        random_state: Seed for reproducibility.
        stratify: If True, stratify the split on ``y``. Only meaningful for
            classification targets with a small number of classes; ignored
            (must be left False) for continuous regression targets.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    strat_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat_arg
    )
    logger.info(
        "Train/test split -> train=%d rows, test=%d rows (test_size=%.2f).",
        len(X_train),
        len(X_test),
        test_size,
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# Preprocessing pipelines
# --------------------------------------------------------------------------


def build_numerical_pipeline() -> Pipeline:
    """Build the numerical preprocessing sub-pipeline.

    Steps, validated in ``03_regression.ipynb`` section 7:

    1. Median imputation -- more robust than the mean to outliers/skewed
       distributions, which are common in this dataset's usage metrics.
    2. ``StandardScaler`` -- puts all numerical features on a comparable
       scale so regularized linear models are not dominated by columns with
       large raw ranges.

    Returns:
        An unfitted scikit-learn ``Pipeline``.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline(min_frequency: float = DEFAULT_MIN_FREQUENCY) -> Pipeline:
    """Build the categorical preprocessing sub-pipeline.

    Steps, validated in ``03_regression.ipynb`` section 7:

    1. Constant imputation with the literal string ``"Missing"`` -- preserves
       missingness as its own signal instead of merging it into the most
       frequent category (relevant for ``sleep_disruption_from_phone``,
       ~27.7% missing).
    2. ``OneHotEncoder`` with ``handle_unknown="ignore"`` (never raises on a
       category unseen during training, e.g. at inference time) and
       ``min_frequency`` (groups rare categories into a single infrequent
       bucket, controlling dimensionality blow-up for high-cardinality
       columns such as ``app_name``, 94 unique values).

    Args:
        min_frequency: Minimum frequency (as a fraction of rows) below which
            a category is grouped into the encoder's infrequent bucket.

    Returns:
        An unfitted scikit-learn ``Pipeline``.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse_output=False),
            ),
        ]
    )


def build_preprocessing_pipeline(
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    min_frequency: float = DEFAULT_MIN_FREQUENCY,
) -> ColumnTransformer:
    """Assemble the full, reusable preprocessing ``ColumnTransformer``.

    This is the single preprocessing entry point intended for reuse by
    classification, clustering, the FastAPI service, the Streamlit app and
    batch inference jobs -- every consumer should build its transformer
    through this function rather than re-implementing imputation, scaling,
    or encoding logic locally.

    Args:
        numerical_features: Columns routed through
            :func:`build_numerical_pipeline`.
        categorical_features: Columns routed through
            :func:`build_categorical_pipeline`.
        min_frequency: Forwarded to :func:`build_categorical_pipeline`.

    Returns:
        An unfitted ``ColumnTransformer``. Call ``.fit`` (or ``fit_transform``)
        only on training data to avoid leaking test/inference-time statistics
        into the fitted imputers, scaler, and encoder.

    Raises:
        ValueError: If both feature lists are empty.
    """
    if not numerical_features and not categorical_features:
        raise ValueError(
            "At least one of numerical_features or categorical_features must be non-empty."
        )

    transformers = []
    if numerical_features:
        transformers.append(("numeric", build_numerical_pipeline(), list(numerical_features)))
    if categorical_features:
        transformers.append(
            ("categorical", build_categorical_pipeline(min_frequency), list(categorical_features))
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


# --------------------------------------------------------------------------
# End-to-end orchestrator
# --------------------------------------------------------------------------


@dataclass
class PreparedDataset:
    """Bundle of everything a model-training step needs to get started.

    Attributes:
        X_train: Training feature matrix, raw (before preprocessing).
        X_test: Test feature matrix, raw (before preprocessing).
        y_train: Training target vector.
        y_test: Test target vector.
        preprocessor: Unfitted ``ColumnTransformer`` built from the training
            columns; call ``preprocessor.fit(X_train)`` (or
            ``fit_transform``) before use -- never fit it on ``X_test``.
        numerical_features: Numerical columns detected in ``X_train``.
        categorical_features: Categorical columns detected in ``X_train``.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    numerical_features: List[str]
    categorical_features: List[str]


def prepare_dataset(
    target: str,
    filename: str = DEFAULT_DATA_FILENAME,
    processed_dir: Optional[Path] = None,
    extra_exclude: Optional[Sequence[str]] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = False,
    min_frequency: float = DEFAULT_MIN_FREQUENCY,
) -> PreparedDataset:
    """End-to-end preprocessing entry point: load -> split -> detect -> build.

    This is the single function future consumers (classification, clustering,
    the FastAPI service, the Streamlit app, batch inference) should call to
    go from the processed dataset on disk to a train/test split plus a
    ready-to-fit preprocessing pipeline, without re-implementing any step
    validated in the notebooks.

    Args:
        target: Name of the target column to predict (e.g.
            ``"screen_time_hours"`` for regression or ``"target"`` for the
            upcoming classification task).
        filename: Processed dataset filename, see :func:`load_processed_data`.
        processed_dir: Optional override for the processed data directory.
        extra_exclude: Additional feature columns to exclude beyond the
            defaults resolved by :func:`get_exclude_columns`.
        test_size: Forwarded to :func:`split_train_test`.
        random_state: Forwarded to :func:`split_train_test`.
        stratify: Forwarded to :func:`split_train_test` (set True for
            classification targets).
        min_frequency: Forwarded to :func:`build_preprocessing_pipeline`.

    Returns:
        A :class:`PreparedDataset` with the train/test split and an unfitted
        preprocessing ``ColumnTransformer``.
    """
    df = load_processed_data(filename=filename, processed_dir=processed_dir)
    X, y = split_features_target(df, target=target, extra_exclude=extra_exclude)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    numerical_features, categorical_features = detect_feature_types(X_train)
    preprocessor = build_preprocessing_pipeline(
        numerical_features, categorical_features, min_frequency=min_frequency
    )

    return PreparedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )


if __name__ == "__main__":
    # Minimal smoke test for manual verification (`python -m src.data.preprocess`).
    # Not part of the public API and not executed on import.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    prepared = prepare_dataset(target="screen_time_hours")
    prepared.preprocessor.fit(prepared.X_train)
    logger.info(
        "Smoke test OK -> X_train=%s, X_test=%s, numerical=%d, categorical=%d, "
        "transformed_train_shape=%s",
        prepared.X_train.shape,
        prepared.X_test.shape,
        len(prepared.numerical_features),
        len(prepared.categorical_features),
        prepared.preprocessor.transform(prepared.X_train).shape,
    )
