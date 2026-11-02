"""Persistent experiment numbering, logging, and submission versioning."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments"
HISTORY_PATH = EXPERIMENT_DIR / "experiment_history.csv"
LOG_PATH = EXPERIMENT_DIR / "EXPERIMENT_LOG.md"
SUBMISSION_DIR = ROOT / "submissions"
HISTORY_FIELDS = [
    "experiment_id",
    "run_at",
    "model",
    "parameters",
    "mean_rmse",
    "std_rmse",
    "fold_rmse",
    "improvement_vs_baseline",
    "observation",
    "submission_file",
]


def _next_experiment_id() -> str:
    if not HISTORY_PATH.exists():
        return "E00"
    history = pd.read_csv(HISTORY_PATH)
    if history.empty:
        return "E00"
    numbers = history["experiment_id"].str.extract(r"E(\d+)")[0].dropna().astype(int)
    return f"E{numbers.max() + 1:02d}"


def _baseline_rmse() -> float | None:
    if not HISTORY_PATH.exists():
        return None
    history = pd.read_csv(HISTORY_PATH)
    baseline = history.loc[history["experiment_id"] == "E00", "mean_rmse"]
    return None if baseline.empty else float(baseline.iloc[0])


def _append_history(row: dict[str, Any]) -> None:
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    write_header = not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_markdown(row: dict[str, Any]) -> None:
    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "# Restaurant Turnover Experiment Log\n\n"
            "Lower cross-validation RMSE is better. Experiments use the fixed "
            "five-fold split with `random_state=42`.\n",
            encoding="utf-8",
        )
    parameters = json.loads(row["parameters"])
    parameter_text = ", ".join(f"`{key}={value}`" for key, value in parameters.items())
    improvement = float(row["improvement_vs_baseline"])
    direction = "better" if improvement >= 0 else "worse"
    entry = (
        f"\n## {row['experiment_id']} — {row['model']}\n\n"
        f"- Run at: {row['run_at']}\n"
        f"- Parameters: {parameter_text}\n"
        f"- Mean RMSE: **{float(row['mean_rmse']):,.0f}**\n"
        f"- Standard deviation: **{float(row['std_rmse']):,.0f}**\n"
        f"- Difference from E00: **{abs(improvement):,.0f} {direction}**\n"
        f"- Fold RMSE: {row['fold_rmse']}\n"
        f"- Submission: `{row['submission_file']}`\n"
        f"- Observation: {row['observation']}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(entry)


def save_experiment(
    *,
    model_name: str,
    parameters: dict[str, Any],
    fold_rmse: Iterable[float],
    predictions: Iterable[float],
    registration_numbers: Iterable[Any],
    observation: str,
    target_name: str = "Annual Turnover",
    id_name: str = "Registration Number",
) -> dict[str, Any]:
    """Assign an ID, save its submission, and append both history files."""
    scores = np.asarray(list(fold_rmse), dtype=float)
    mean_rmse = float(scores.mean())
    std_rmse = float(scores.std())
    experiment_id = _next_experiment_id()
    baseline = _baseline_rmse()
    improvement = 0.0 if baseline is None else baseline - mean_rmse

    SUBMISSION_DIR.mkdir(exist_ok=True)
    filename = f"submission_{experiment_id}_rmse_{mean_rmse:.0f}.csv"
    submission_path = SUBMISSION_DIR / filename
    submission = pd.DataFrame(
        {id_name: list(registration_numbers), target_name: list(predictions)}
    )
    if submission.shape != (500, 2) or submission.isna().any().any():
        raise ValueError("Submission must contain 500 complete rows and two columns")
    if (submission[target_name] < 0).any():
        raise ValueError("Submission contains negative turnover predictions")
    submission.to_csv(submission_path, index=False)

    row = {
        "experiment_id": experiment_id,
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model_name,
        "parameters": json.dumps(parameters, sort_keys=True),
        "mean_rmse": round(mean_rmse, 6),
        "std_rmse": round(std_rmse, 6),
        "fold_rmse": json.dumps([round(value, 6) for value in scores]),
        "improvement_vs_baseline": round(improvement, 6),
        "observation": observation.strip() or "No observation supplied.",
        "submission_file": filename,
    }
    _append_history(row)
    _append_markdown(row)
    return {**row, "submission_path": submission_path, "submission": submission}

