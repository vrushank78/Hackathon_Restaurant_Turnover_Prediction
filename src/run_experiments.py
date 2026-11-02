"""Run a small, reproducible Extra Trees experiment set."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from train import ID_COLUMN, RANDOM_STATE, TARGET, add_date_features, build_pipeline


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    {"experiment_id": "E00", "min_samples_leaf": 2, "max_features": 0.8, "max_depth": None},
    {"experiment_id": "E01", "min_samples_leaf": 1, "max_features": 0.8, "max_depth": None},
    {"experiment_id": "E02", "min_samples_leaf": 4, "max_features": 0.8, "max_depth": None},
    {"experiment_id": "E03", "min_samples_leaf": 8, "max_features": 0.8, "max_depth": None},
    {"experiment_id": "E04", "min_samples_leaf": 2, "max_features": 0.5, "max_depth": None},
    {"experiment_id": "E05", "min_samples_leaf": 2, "max_features": 0.7, "max_depth": None},
    {"experiment_id": "E06", "min_samples_leaf": 2, "max_features": 1.0, "max_depth": None},
    {"experiment_id": "E07", "min_samples_leaf": 2, "max_features": "sqrt", "max_depth": None},
    {"experiment_id": "E08", "min_samples_leaf": 2, "max_features": 0.8, "max_depth": 20},
]


def main() -> None:
    train = pd.read_csv(ROOT / "Train_dataset.csv", na_values=["NA"])
    X = add_date_features(train.drop(columns=[TARGET, ID_COLUMN]))
    y = train[TARGET].astype(float)
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for config in EXPERIMENTS:
        pipeline = build_pipeline(X)
        pipeline.set_params(
            model__min_samples_leaf=config["min_samples_leaf"],
            model__max_features=config["max_features"],
            model__max_depth=config["max_depth"],
        )
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=folds,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        rmse = np.sqrt(-scores)
        row = {
            **config,
            "mean_rmse": rmse.mean(),
            "std_rmse": rmse.std(),
            **{f"fold_{i}_rmse": score for i, score in enumerate(rmse, start=1)},
        }
        rows.append(row)
        print(
            f"{config['experiment_id']}: mean={rmse.mean():,.0f}, "
            f"std={rmse.std():,.0f}"
        )

    results = pd.DataFrame(rows).sort_values("mean_rmse")
    output = ROOT / "models" / "experiment_results.csv"
    output.parent.mkdir(exist_ok=True)
    results.to_csv(output, index=False)
    print("\nRanked results:\n", results.to_string(index=False))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
