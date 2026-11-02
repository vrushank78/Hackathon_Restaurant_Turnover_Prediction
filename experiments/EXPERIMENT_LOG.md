# Restaurant Turnover Experiment Log

Lower cross-validation RMSE is better. All experiments use the same five shuffled folds with `random_state=42`.

## Existing experiments E00–E08

- **E00:** baseline; mean RMSE **20,218,712**. Trustworthy starting point.
- **E01:** leaf 1; mean RMSE **20,715,176**. Lower variation, but worse mean; rejected.
- **E02:** leaf 4; mean RMSE **20,202,836**. Tiny improvement.
- **E03:** leaf 8; mean RMSE **20,065,181**. Smoothing helped.
- **E04:** feature fraction 0.5; mean RMSE **20,052,211**. Strong runner-up.
- **E05:** feature fraction 0.7; mean RMSE **20,134,932**. Better than baseline.
- **E06:** all features; mean RMSE **20,505,073**. Worse; rejected.
- **E07:** square-root features; mean RMSE **20,031,412**. Best of the first batch.
- **E08:** depth 20; mean RMSE **20,257,594**. Slightly worse than unlimited depth.
- **E09:** square-root features with depth 35; mean RMSE **20,018,521**. Best result in the user's subsequent notebook run.

New runs are appended below automatically by `save_experiment()`.


## E10 — ExtraTreesRegressor

- Run at: 2026-11-02T17:12:16-06:00
- Parameters: `max_depth=35`, `max_features=sqrt`, `min_samples_leaf=3`, `n_estimators=400`
- Mean RMSE: **20,060,530**
- Standard deviation: **2,694,306**
- Difference from E00: **158,182 better**
- Fold RMSE: [19563820.127694, 22512551.889555, 20606615.748509, 15162188.084437, 22457473.975444]
- Submission: `submission_E10_rmse_20060530.csv`
- Observation: Testing a slightly larger leaf with square-root feature sampling and depth 35.

## E11 — ExtraTreesRegressor

- Run at: 2026-11-02T17:18:53-06:00
- Parameters: `max_depth=40`, `max_features=0.5`, `min_samples_leaf=3`, `n_estimators=800`
- Mean RMSE: **20,084,967**
- Standard deviation: **2,608,536**
- Difference from E00: **133,745 better**
- Fold RMSE: [19492358.705983, 22634739.278107, 20464834.658391, 15431152.684961, 22401751.47256]
- Submission: `submission_E11_rmse_20084967.csv`
- Observation: Testing a slightly larger leaf with square-root feature sampling and depth 35.

## E12 — ExtraTreesRegressor

- Run at: 2026-11-02T17:21:39-06:00
- Parameters: `max_depth=30`, `max_features=0.6`, `min_samples_leaf=6`, `n_estimators=600`
- Mean RMSE: **20,012,881**
- Standard deviation: **2,676,180**
- Difference from E00: **205,831 better**
- Fold RMSE: [19266491.2945, 22657698.125352, 20437722.217144, 15289066.460893, 22413427.079415]
- Submission: `submission_E12_rmse_20012881.csv`
- Observation: Testing a slightly larger leaf with square-root feature sampling and depth 35.
