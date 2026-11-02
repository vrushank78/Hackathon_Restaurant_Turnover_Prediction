# Restaurant Annual Turnover Prediction

Beginner-friendly hackathon project for predicting restaurant annual turnover. The competition metric is **RMSE** (lower is better).

## Project layout

```text
Hackathon/
├── Business_Context.txt        # Original problem statement
├── Train_dataset.csv           # 3,493 labelled rows
├── Test_dataset.csv            # 500 unlabelled rows
├── Sample_Solution.csv         # Required submission shape
├── Hackathon_Sample_Code.ipynb # Original sample notebook
├── notebooks/                  # Your experiments and notes
├── src/train.py                # Reproducible baseline model
├── models/                     # Generated models (not committed)
└── submissions/                # Generated submissions (not committed)
```

## 1. Set up once (PowerShell)

Run these commands from this `Hackathon` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` in place of `python` for later commands.

## 2. Run the baseline

```powershell
python src/train.py
```

This script:

- keeps `Registration Number` only for the final submission;
- converts the opening date into restaurant age and calendar features;
- fills missing numeric and categorical values;
- one-hot encodes categories;
- reports 5-fold cross-validation RMSE;
- trains on all labelled rows; and
- writes `submissions/baseline_submission.csv` with exactly the required two columns.

## 3. Recommended hackathon workflow

1. **Understand the goal:** predict `Annual Turnover`; optimize cross-validation RMSE, not training score.
2. **Audit the data:** inspect data types, missingness, target distribution, unusual values, and train/test category differences.
3. **Lock a baseline:** run `src/train.py` unchanged and record its mean and fold RMSE values.
4. **Experiment one change at a time:** feature engineering, model choice, or hyperparameters. Record every result.
5. **Prevent leakage:** never use `Annual Turnover` as an input and never tune using the test set.
6. **Validate consistently:** keep the same folds while comparing experiments. A lower, stable CV RMSE is the signal to trust.
7. **Submit carefully:** check row count, columns, order, missing predictions, and non-negative turnover.

## Running a new tracked experiment

Open `notebooks/Restaurant_Turnover_Prediction.ipynb`. In the validation section:

1. Change the values in `EXPERIMENT_PARAMETERS`.
2. Update `EXPERIMENT_OBSERVATION` to explain what you are testing.
3. Run from that validation cell through the save cell once.

The tracker automatically assigns the next ID, saves a file such as
`submissions/submission_E10_rmse_20060530.csv`, appends the structured result to
`experiments/experiment_history.csv`, and appends a readable entry to
`experiments/EXPERIMENT_LOG.md`. Existing results are never overwritten.

Good early experiments are: log-transforming the skewed target, extracting restaurant age, grouping rare categories, and comparing Random Forest/Extra Trees/Gradient Boosting models. Advanced libraries such as CatBoost can be tried later, after the baseline is trustworthy.

## Suggested schedule

- **Preparation:** run the baseline, understand every column, and create an experiment log.
- **Exploration:** investigate missing values, target skew, suspicious outliers, and useful relationships.
- **Modeling:** compare a few models with the same cross-validation folds.
- **Final day:** rerun the best pipeline from scratch, validate the submission, and keep time for upload issues.

## Git routine

Commit small milestones instead of every output file:

```powershell
git status
git add README.md requirements.txt .gitignore src notebooks models submissions
git commit -m "Set up restaurant turnover hackathon project"
git push
```

Do not commit `.venv`, trained models, or generated submissions. The original datasets are small and are currently intended to be versioned; remove them from Git later if the competition rules prohibit sharing data.

