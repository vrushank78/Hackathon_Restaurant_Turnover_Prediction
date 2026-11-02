"""Train a reproducible baseline and create a competition submission."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "Train_dataset.csv"
TEST_PATH = ROOT / "Test_dataset.csv"
TARGET = "Annual Turnover"
ID_COLUMN = "Registration Number"
RANDOM_STATE = 42


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn the opening-date text into simple numeric features."""
    result = frame.copy()
    date_column = "Opening Day of Restaurant"
    opened = pd.to_datetime(result.pop(date_column), dayfirst=True, errors="coerce")
    result["Opening Year"] = opened.dt.year
    result["Opening Month"] = opened.dt.month
    result["Restaurant Age"] = 2026 - opened.dt.year
    return result


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    model = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def main() -> None:
    train = pd.read_csv(TRAIN_PATH, na_values=["NA"])
    test = pd.read_csv(TEST_PATH, na_values=["NA"])

    test_ids = test[ID_COLUMN].copy()
    y = train[TARGET].astype(float)
    X = add_date_features(train.drop(columns=[TARGET, ID_COLUMN]))
    X_test = add_date_features(test.drop(columns=[ID_COLUMN]))
    # Competition files are not perfectly symmetric: for example, the supplied
    # test set omits "Endorsed By". Add any absent training columns as missing
    # values and discard/reorder unexpected columns before preprocessing.
    X_test = X_test.reindex(columns=X.columns)

    pipeline = build_pipeline(X)
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    negative_mse = cross_val_score(
        pipeline, X, y, cv=folds, scoring="neg_mean_squared_error", n_jobs=1
    )
    fold_rmse = np.sqrt(-negative_mse)
    print("Fold RMSE:", ", ".join(f"{score:,.0f}" for score in fold_rmse))
    print(f"Mean RMSE: {fold_rmse.mean():,.0f} (+/- {fold_rmse.std():,.0f})")

    pipeline.fit(X, y)
    predictions = np.clip(pipeline.predict(X_test), a_min=0, a_max=None)

    submission = pd.DataFrame({ID_COLUMN: test_ids, TARGET: predictions})
    if submission.shape != (500, 2) or submission.isna().any().any():
        raise ValueError("Submission validation failed")

    submissions_dir = ROOT / "submissions"
    models_dir = ROOT / "models"
    submissions_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    submission.to_csv(submissions_dir / "baseline_submission.csv", index=False)
    joblib.dump(pipeline, models_dir / "baseline_pipeline.joblib")
    print(f"Saved: {submissions_dir / 'baseline_submission.csv'}")


if __name__ == "__main__":
    main()

