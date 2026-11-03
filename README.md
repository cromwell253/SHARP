# SHARP

History-selective attention and prediction-guided spatial incremental correction
for multi-node, multi-step mining-induced stress forecasting.

## Contents

- `data/`: seven experimental inputs, with stress readings and local coordinates.
- `sharp/models/`: the deployed architecture and its components.
- `checkpoints/`: all seven datasets, seeds 2026/2027/2028, and five ablation configurations.
- `evaluate.py`: frozen-checkpoint evaluation on chronological splits.
- `replay.py`: one-minute historical replay under training-only quantile thresholds.
- `sharp/training.py`: reusable training kernels for the two correction stages.
- `configs/`: splits, normalization, geometry partitions, graph inputs, stage settings and recorded epochs.
- `results/comparison.xlsx`, `results/ablation.xlsx`: seven-dataset result tables.
- `results/comparison/`: per-seed, per-dataset and across-dataset comparison results.
- `results/engineering/`: five-minute accuracy, threshold replay, input perturbations and latency measurements.

## Install and evaluate

Python 3.10 was used. Install a PyTorch build appropriate for your device; the
recorded evaluation environment used PyTorch 2.3.0 with CUDA 11.8.

```bash
pip install -r requirements.txt
python evaluate.py --dataset JAPD --seed 2026 --variant SHARP --device cpu
python evaluate.py --dataset WF1 --seed 2026 --variant PatchSTG --device cuda
python replay.py --dataset JAPD --seed 2026 --variant SHARP --device cuda
```

`--limit 32` provides a short execution check. Full evaluation omits that option.
Generated output goes to ignored `runs/` files and does not replace paper results.

```python
import numpy as np
from sharp.inference import load_model

model = load_model('.', dataset='WF1', seed=2026, device='cpu')
data = np.load('data/WF1/series.npz', allow_pickle=False)['data']
forecast = model.predict(data[:12][None])  # [1,48,36], MPa
```

## Data

The released arrays are the inputs recorded in the final comparison manifest:
JAPD, WF1, WF13112, WF15212, WF1905, WF2301 and WF2903. Each directory contains
only `series.npz` and `coordinates.csv`. The NPZ contains a single numeric `data`
array shaped `[time,nodes]`, in experimental float32 precision and MPa units.
The sampling interval is one minute. Calendar dates and original sensor labels
are removed; row and column order are retained. Coordinates use the first sensor
as the local origin, preserving pairwise distances. Experimental graph inputs
and partitions are in `configs/`, because they are needed for reconstruction.

Stress readings were checked for equality before and after export. This release
does not add measurement noise or rescale stress values. Array-content hashes,
split boundaries and window counts are in `configs/datasets.json`.

## Experiment protocol

Inputs comprise 12 minutes; each call outputs 48 minutes for every node. Raw
periods are split chronologically 60/20/20. Every target lies within its split;
validation/test input may use preceding observations. Training/evaluation window
starts are spaced by five minutes; event replay uses one-minute starts.
Normalization uses one mean and standard deviation from the training period.
Time covariates are relative history positions 0..11 and weekday index zero.

MAE and RMSE are in physical stress units; lower is better. Leads 12,24,48 mean
the exact forecast positions. Overall metrics include all 48 positions. Main
tables average three seeds within each dataset. Across-dataset SD describes
variation across seven dataset means, not variation across seeds.

See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for stage training, ablations, engineering
metrics and reproduction scope. All reported configurations are retained,
including cases where adding a component increases error.

## Results and reproduction scope

The main comparison contains SHARP and 19 independent baselines. Duplicate or
retired entries are excluded and all rankings are recomputed. The matched
PatchSTG checkpoint used by ablation/engineering tests has its own documented
configuration; it is not a guarantee that every historical PatchSTG run has
identical weights or architecture settings.

This package supports inference and evaluation of the released SHARP, matched
PatchSTG and ablation checkpoints without the original research workspace.
`results/checkpoint_reload_checks.json` records export/reload prediction checks
for 105 cases. Historical third-party baseline metrics are provided as archived
results; those models were not retrained or independently revalidated during
packaging, and their complete source projects are not vendored here.

The correction-stage training kernels and settings are provided with the frozen
base weights. They are not a turnkey reconstruction of every upstream backbone
training run or the original model development/search process. Stage logs and
checkpoint evaluation should be used to distinguish these reproduction scopes.

Floating-point reduction order, device and library versions can cause small
evaluation differences. Archived paper values remain available for comparison.

## Engineering evaluation

Five-minute accuracy uses all three seeds. The archived minute-wise threshold
replay and input perturbations use seed 2026. Replay covers upper quantiles
0.90/0.95/0.99 estimated only from training data. These are experimental scenarios,
not mine-approved alarm settings or accident labels. Event recall counts distinct
observed threshold-entry episodes; window precision is a separate measure.

Missing-input and two-minute-stale-input evaluations already cover all seven
datasets for SHARP and matched PatchSTG, with the same perturbations per pair.
Only two example datasets were shown in the manuscript robustness table.
The archived random-missing test uses one mask seed, not repeated independent masks.

Latency includes local preprocessing, inference and result serialization. It
does not include mine sensor/network transmission or a control-room display.
The results are offline laboratory measurements, not evidence of field deployment.

## Attribution

The PatchSTG-derived backbone retains its MIT notice in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Licensing for the project additions
and monitoring data has not yet been separately specified.

## Extended engineering evaluation

`results/engineering_extension/` adds three model seeds, three thresholds,
three independent missing masks, 1,000-update accelerated software replay, and
transparent alert cases. Run `python experiments/engineering_extension.py` to
recompute using released checkpoints (CUDA required). Plotting requires matplotlib.
The updated replay entry point preserves the original first/last stride-five
test-window bounds when constructing one-minute updates.

The new episode protocol merges consecutive pre-onset alarms and matches only
the first issue of an alarm episode to at most one onset within five minutes.
Its event precision/recall/F1 are distinct from the retained any-window event
recall. Across seven datasets and three seeds at q95, the new SHARP event F1 is
about 0.264, compared with 0.352 for matched PatchSTG. This is a negative outcome,
despite lower continuous errors. No threshold was adjusted on the test period.

The displayed alert-hit case is not a successful full-curve fit. The plotted
SHARP output exactly matches the original model; both selected examples fail
to follow parts of the subsequent 48-minute trajectory. The diagnostic audits
and fixed-error-quantile WF1 examples are released alongside the curves.
