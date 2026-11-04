# Engineering evaluation

This folder contains the protocols, runnable experiments and result tables for
five-minute forecasting, threshold-event warnings, update latency and incomplete
inputs. It uses the seven published datasets and the supplied frozen checkpoints.

## Warning calibration

Each sensor's stress threshold is a fixed 90th, 95th or 99th training percentile.
These are evaluation scenarios; the primary scenario is the 95th percentile.
SHARP and PatchSTG receive the same validation-only calibration procedure.
The chronological validation split is divided in half: the first half fits
coefficients, with five-step targets contained before the midpoint; the second
half selects operating thresholds. A common calibration family is selected by
mean validation event F1 across both models, all datasets and three seeds.
The four candidates and fixed hyperparameters are in
[`configs/warning_calibration_protocol.json`](configs/warning_calibration_protocol.json).

The selected family, `bias_margin`, scores the predicted five-minute maximum
relative to the fixed stress threshold and training-node standard deviation,
then adds a node-wise validation bias. The bias is shrunk to its cross-node mean
with 256 pseudo-samples. The score threshold is selected either to maximize
validation event F1 or subject to a validation false-alarm budget. If the
threshold-selection half has no observed onsets, the primary operating point
retains direct thresholding for both models. All choices are fixed before
calibrated test reassessment. This evaluation reuses the previously examined
chronological test split, rather than a newly collected external test set.
The continuous forecasts and forecasting error tables are unchanged.

Consecutive node alerts form an episode. Its first issue is matched to at most
one observed onset in the next five minutes; each onset can be matched once.
Only current readings at or below the stress threshold can issue pre-onset
alerts. Eligible onsets require five complete preceding issue opportunities.
Precision, recall and F1 are higher-is-better. FA is unmatched alert episodes
per sensor-hour and is lower-is-better. Lead time is observed onset minus the
first matched issue, averaged over detected events. Each dataset averages its
three seeds before the seven datasets receive equal weight.

| q95 operating point | Model | Precision (%) | Recall (%) | F1 (%) | FA / sensor-hour | Lead (min) |
|---|---|---:|---:|---:|---:|---:|
| Validation F1 maximum | SHARP | 53.81 | 72.72 | 60.44 | 0.0340 | 2.21 |
| Validation F1 maximum | PatchSTG | 52.62 | 65.05 | 57.18 | 0.0300 | 1.87 |
| Validation FA <= 0.02 | SHARP | 56.93 | 34.32 | 41.19 | 0.0114 | 1.86 |
| Validation FA <= 0.02 | PatchSTG | 53.08 | 33.50 | 39.20 | 0.0132 | 1.88 |

At the F1 operating point, all seven dataset means favor SHARP. The mean F1
difference is 3.27 percentage points; a paired dataset bootstrap (20,000 draws,
RNG seed 20261104) gives a 95% interval of [1.67, 5.84] points. SHARP has higher
recall and mean lead, alongside a higher unmatched-alert frequency. The FA
constraint is applied on validation, not an assertion of identical test FA.
The original direct-threshold results remain in `event_metrics.csv` and in the
new case table. History-only logistic references are also included; their
repeated model labels refer to the same history-only detector.

## Results and parameters

| File in `results/` | Contents |
|---|---|
| `test_case_metrics.csv` | Seven datasets, three seeds, two forecasters, three quantiles; direct, calibrated and history-only event metrics |
| `test_macro_summary.csv` | Equal-dataset summary for every published operating point |
| `paired_effect_intervals.csv` | Paired dataset bootstrap differences and intervals |
| `validation_family_summary.csv` | Common-family selection scores |
| `event_metrics.csv` | Original uncalibrated one-to-one event results |
| `forecast_metrics.csv` | Five-minute numerical errors |
| `robustness_metrics.csv` | Missingness and stale-input results |
| `runtime_summary.csv`, `cpu_runtime_summary.csv` | GPU and CPU processing latency |
| `continuous_pipeline_summary.json`, `visual_pipeline_summary.json` | Sequential processing and local browser interface timing |
| `spatial_correlation_diagnostic.csv` | Descriptive sensor correlations for the discussion |

[`configs/warning_operating_points.json`](configs/warning_operating_points.json)
stores every selected bias, scale, stress threshold and score threshold as plain
JSON. The direct event protocol and timing environment are separate config files.

## Other evaluation settings

Missingness removes 10% of historical readings, preserves the first input
position, and fills each missing value from the preceding available value.
Mask seeds are 20261101, 20261102 and 20261103; both forecasters receive the same
mask. Staleness replaces the latest two readings with the preceding reading.
The formal robustness table covers all seven datasets and three model seeds.

Processing latency uses batch one, 50 warmups and 200 measured windows per
group, with GPU synchronization and two PyTorch CPU threads. It includes
normalization, inference, inverse transformation and result serialization.
P99 is the empirical 99th percentile. The separate WF1 continuous run measures
1,000 accelerated updates per model. The browser prototype adds input/output
HTTP and interface drawing, with 50 inference and 10 interface warmup updates.

Spatial diagnostics describe the final 20% of each dataset. Geometric
four-neighbor pairs are deduplicated and compared with the other sensor pairs;
stress levels and one-minute increments are summarized separately. Undefined
zero-variance correlations are excluded, with valid pair counts retained.

## Reproduction

Run commands from the repository root. The recorded environment uses Python 3.10
and PyTorch 2.3.0; engineering-only Python dependencies are listed separately:

```bash
pip install -r engineering/requirements.txt
python engineering/experiments/calibrated_warning_experiment.py --repo . --out runs/warning_calibration --stage infer
python engineering/experiments/calibrated_warning_experiment.py --repo . --out runs/warning_calibration --stage select
python engineering/experiments/calibrated_warning_experiment.py --repo . --out runs/warning_calibration --stage test
```

The study uses batch 128 for validation inference and batch 32 for test inference,
two PyTorch CPU threads, disabled CUDA matmul TF32 and enabled cuDNN TF32. It
generates per-run caches in the ignored `runs/` folder. The main model loader
selects CUDA when available. Small numerical changes across hardware or library
versions can affect threshold ties.

For inference using the supplied settings, call `calibrated_warning` from
`engineering/experiments/apply_calibrated_warning.py` with the current readings
and a forecast of shape `[batch, 48, nodes]`. The returned array is a node-wise
boolean alert mask. Both models use the same scoring equation and their own
validation-fitted parameters.

Other evaluations:

```bash
python engineering/experiments/engineering_extension.py
python engineering/experiments/spatial_diagnostic.py --repo . --output runs/spatial_correlation_diagnostic.csv
python engineering/experiments/ui_monitor_backend.py --repo . --out runs/ui_validation
```

With the UI backend running, launch the client in a second terminal with Node.js
and Playwright installed:

```bash
node engineering/experiments/ui_monitor_client.cjs runs/ui_validation
```

Windows uses installed Microsoft Edge; other systems use Playwright Chromium.
The recorded WF1 test uses 1,000 sequential minute-indexed updates per model in
accelerated wall time. Timing includes local HTTP, model inference, JSON output,
canvas drawing and two animation-frame callbacks. This measures a local browser
prototype; the original processing-only timing is reported separately.
